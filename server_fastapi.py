"""
API server for TTS
TODO: server_editor.pyと統合する?
"""

import argparse
import os
import sys
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Any, Optional
from urllib.parse import unquote

import GPUtil
import psutil
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request, UploadFile, File, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from scipy.io import wavfile

from config import get_config
from style_bert_vits2.constants import (
    DEFAULT_ASSIST_TEXT_WEIGHT,
    DEFAULT_LENGTH,
    DEFAULT_LINE_SPLIT,
    DEFAULT_NOISE,
    DEFAULT_NOISEW,
    DEFAULT_SDP_RATIO,
    DEFAULT_SPLIT_INTERVAL,
    DEFAULT_STYLE,
    DEFAULT_STYLE_WEIGHT,
    Languages,
)
from style_bert_vits2.logging import logger
from style_bert_vits2.nlp import bert_models, onnx_bert_models
from style_bert_vits2.nlp.japanese import pyopenjtalk_worker as pyopenjtalk
from style_bert_vits2.nlp.japanese.user_dict import update_dict
from style_bert_vits2.tts_model import TTSModel, TTSModelHolder
from style_bert_vits2.utils import torch_device_to_onnx_providers
from transcribe import transcribe_with_faster_whisper


config = get_config()
ln = config.server_config.language


# pyopenjtalk_worker を起動
## pyopenjtalk_worker は TCP ソケットサーバーのため、ここで起動する
pyopenjtalk.initialize_worker()

# dict_data/ 以下の辞書データを pyopenjtalk に適用
update_dict()


def raise_validation_error(msg: str, param: str):
    logger.warning(f"Validation error: {msg}")
    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail=[dict(type="invalid_params", msg=msg, loc=["query", param])],
    )


class AudioResponse(Response):
    media_type = "audio/wav"


loaded_models: list[TTSModel] = []

# Whisper モデル (音声認識用)
# 起動時に初期化され、/voice_changer エンドポイントで使用される
whisper_model: Optional[Any] = None


def load_models(model_holder: TTSModelHolder):
    global loaded_models
    loaded_models = []
    for model_name, model_paths in model_holder.model_files_dict.items():
        model = TTSModel(
            model_path=model_paths[0],
            config_path=model_holder.root_dir / model_name / "config.json",
            style_vec_path=model_holder.root_dir / model_name / "style_vectors.npy",
            device=model_holder.device,
        )
        # 起動時に全てのモデルを読み込むのは時間がかかりメモリを食うのでやめる
        # model.load()
        loaded_models.append(model)


def load_whisper_model(device: str = "cpu", model_size: str = "large-v3"):
    """
    Whisper モデルを初期化する (音声認識用)

    Args:
        device (str): 使用するデバイス ("cpu" or "cuda")
        model_size (str): Whisper モデルのサイズ ("large-v3", "large-v2", "large" など)
    """
    global whisper_model

    try:
        from faster_whisper import WhisperModel

        logger.info(f"Loading Whisper model ({model_size}) on {device}...")

        # compute_type の自動選択
        if device == "cuda":
            compute_type = "float16"
        else:
            compute_type = "int8"

        try:
            whisper_model = WhisperModel(model_size, device=device, compute_type=compute_type)
            logger.info(f"Whisper model loaded successfully on {device}")
        except ValueError as e:
            logger.warning(f"Failed to load Whisper model with compute_type={compute_type}, trying auto: {e}")
            whisper_model = WhisperModel(model_size, device=device)
            logger.info(f"Whisper model loaded successfully with auto compute_type")

    except ImportError:
        logger.warning("faster-whisper is not installed. Voice changer endpoint will not be available.")
        whisper_model = None
    except Exception as e:
        logger.error(f"Failed to load Whisper model: {e}")
        whisper_model = None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true", help="Use CPU instead of GPU")
    parser.add_argument(
        "--dir", "-d", type=str, help="Model directory", default=config.assets_root
    )
    parser.add_argument("--preload_onnx_bert", action="store_true")
    args = parser.parse_args()

    if args.cpu:
        device = "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # 事前に BERT モデル/トークナイザーをロードしておく
    ## ここでロードしなくても必要になった際に自動ロードされるが、時間がかかるため事前にロードしておいた方が体験が良い
    ## 英語や中国語で音声合成するユースケースは限られていることから、VRAM 節約のため日本語の BERT モデル/トークナイザーのみロードする
    bert_models.load_model(Languages.JP, device_map=device)
    bert_models.load_tokenizer(Languages.JP)
    # VRAM 節約のため、既定では ONNX 版 BERT モデル/トークナイザーは事前ロードしない
    if args.preload_onnx_bert:
        onnx_bert_models.load_model(
            Languages.JP, onnx_providers=torch_device_to_onnx_providers(device)
        )
        onnx_bert_models.load_tokenizer(Languages.JP)

    model_dir = Path(args.dir)
    model_holder = TTSModelHolder(
        model_dir, device, torch_device_to_onnx_providers(device)
    )
    if len(model_holder.model_names) == 0:
        logger.error(f"Models not found in {model_dir}.")
        sys.exit(1)

    logger.info("Loading models...")
    load_models(model_holder)

    # Whisper モデルをロード (音声認識用)
    logger.info("Loading Whisper model for voice changer...")
    load_whisper_model(device=device, model_size="large-v3")

    limit = config.server_config.limit
    if limit < 1:
        limit = None
    else:
        logger.info(
            f"The maximum length of the text is {limit}. If you want to change it, modify config.yml. Set limit to -1 to remove the limit."
        )
    app = FastAPI()
    allow_origins = config.server_config.origins
    if allow_origins:
        logger.warning(
            f"CORS allow_origins={config.server_config.origins}. If you don't want, modify config.yml"
        )
        app.add_middleware(
            CORSMiddleware,
            allow_origins=config.server_config.origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    # app.logger = logger
    # ↑効いていなさそう。loggerをどうやって上書きするかはよく分からなかった。

    @app.api_route("/voice", methods=["GET", "POST"], response_class=AudioResponse)
    async def voice(
        request: Request,
        text: str = Query(..., min_length=1, max_length=limit, description="セリフ"),
        encoding: str = Query(None, description="textをURLデコードする(ex, `utf-8`)"),
        model_name: str = Query(
            None,
            description="モデル名(model_idより優先)。model_assets内のディレクトリ名を指定",
        ),
        model_id: int = Query(
            0, description="モデルID。`GET /models/info`のkeyの値を指定ください"
        ),
        speaker_name: str = Query(
            None,
            description="話者名(speaker_idより優先)。esd.listの2列目の文字列を指定",
        ),
        speaker_id: int = Query(
            0, description="話者ID。model_assets>[model]>config.json内のspk2idを確認"
        ),
        sdp_ratio: float = Query(
            DEFAULT_SDP_RATIO,
            description="SDP(Stochastic Duration Predictor)/DP混合比。比率が高くなるほどトーンのばらつきが大きくなる",
        ),
        noise: float = Query(
            DEFAULT_NOISE,
            description="サンプルノイズの割合。大きくするほどランダム性が高まる",
        ),
        noisew: float = Query(
            DEFAULT_NOISEW,
            description="SDPノイズ。大きくするほど発音の間隔にばらつきが出やすくなる",
        ),
        length: float = Query(
            DEFAULT_LENGTH,
            description="話速。基準は1で大きくするほど音声は長くなり読み上げが遅まる",
        ),
        language: Languages = Query(ln, description="textの言語"),
        auto_split: bool = Query(DEFAULT_LINE_SPLIT, description="改行で分けて生成"),
        split_interval: float = Query(
            DEFAULT_SPLIT_INTERVAL, description="分けた場合に挟む無音の長さ（秒）"
        ),
        assist_text: Optional[str] = Query(
            None,
            description="このテキストの読み上げと似た声音・感情になりやすくなる。ただし抑揚やテンポ等が犠牲になる傾向がある",
        ),
        assist_text_weight: float = Query(
            DEFAULT_ASSIST_TEXT_WEIGHT, description="assist_textの強さ"
        ),
        style: Optional[str] = Query(DEFAULT_STYLE, description="スタイル"),
        style_weight: float = Query(DEFAULT_STYLE_WEIGHT, description="スタイルの強さ"),
        reference_audio_path: Optional[str] = Query(
            None, description="スタイルを音声ファイルで行う"
        ),
    ):
        """Infer text to speech(テキストから感情付き音声を生成する)"""
        logger.info(
            f"{request.client.host}:{request.client.port}/voice  { unquote(str(request.query_params) )}"
        )
        if request.method == "GET":
            logger.warning(
                "The GET method is not recommended for this endpoint due to various restrictions. Please use the POST method."
            )
        if model_id >= len(
            model_holder.model_names
        ):  # /models/refresh があるためQuery(le)で表現不可
            raise_validation_error(f"model_id={model_id} not found", "model_id")

        if model_name:
            # load_models() の 処理内容が i の正当性を担保していることに注意
            model_ids = [
                i
                for i, x in enumerate(model_holder.models_info)
                if x.name == model_name
            ]
            if not model_ids:
                raise_validation_error(
                    f"model_name={model_name} not found", "model_name"
                )
            # 今の実装ではディレクトリ名が重複することは無いはずだが...
            if len(model_ids) > 1:
                raise_validation_error(
                    f"model_name={model_name} is ambiguous", "model_name"
                )
            model_id = model_ids[0]

        model = loaded_models[model_id]
        if speaker_name is None:
            if speaker_id not in model.id2spk.keys():
                raise_validation_error(
                    f"speaker_id={speaker_id} not found", "speaker_id"
                )
        else:
            if speaker_name not in model.spk2id.keys():
                raise_validation_error(
                    f"speaker_name={speaker_name} not found", "speaker_name"
                )
            speaker_id = model.spk2id[speaker_name]
        if style not in model.style2id.keys():
            raise_validation_error(f"style={style} not found", "style")
        assert style is not None
        if encoding is not None:
            text = unquote(text, encoding=encoding)
        sr, audio = model.infer(
            text=text,
            language=language,
            speaker_id=speaker_id,
            reference_audio_path=reference_audio_path,
            sdp_ratio=sdp_ratio,
            noise=noise,
            noise_w=noisew,
            length=length,
            line_split=auto_split,
            split_interval=split_interval,
            assist_text=assist_text,
            assist_text_weight=assist_text_weight,
            use_assist_text=bool(assist_text),
            style=style,
            style_weight=style_weight,
        )
        logger.success("Audio data generated and sent successfully")
        with BytesIO() as wavContent:
            wavfile.write(wavContent, sr, audio)
            return Response(content=wavContent.getvalue(), media_type="audio/wav")

    @app.post("/voice_changer", response_class=AudioResponse)
    async def voice_changer(
        request: Request,
        audio_file: UploadFile = File(..., description="変換元の音声ファイル (WAV形式推奨)"),
        model_name: str = Query(
            None,
            description="変換先モデル名(model_idより優先)。model_assets内のディレクトリ名を指定",
        ),
        model_id: int = Query(0, description="変換先モデルID"),
        speaker_name: str = Query(
            None,
            description="変換先話者名(speaker_idより優先)",
        ),
        speaker_id: int = Query(0, description="変換先話者ID"),
        language: str = Query("ja", description="音声認識言語 (ja/en/zh)"),
        whisper_initial_prompt: str = Query(
            "",
            description="Whisper認識の初期プロンプト（認識精度向上に使用）",
        ),
        sdp_ratio: float = Query(DEFAULT_SDP_RATIO, description="SDP/DP混合比"),
        noise: float = Query(DEFAULT_NOISE, description="サンプルノイズの割合"),
        noisew: float = Query(DEFAULT_NOISEW, description="SDPノイズ"),
        length: float = Query(DEFAULT_LENGTH, description="話速"),
        auto_split: bool = Query(DEFAULT_LINE_SPLIT, description="改行で分けて生成"),
        split_interval: float = Query(DEFAULT_SPLIT_INTERVAL, description="分割時の無音の長さ（秒）"),
        assist_text: Optional[str] = Query(None, description="感情表現の参照テキスト"),
        assist_text_weight: float = Query(DEFAULT_ASSIST_TEXT_WEIGHT, description="assist_textの強さ"),
        style: Optional[str] = Query(DEFAULT_STYLE, description="スタイル"),
        style_weight: float = Query(DEFAULT_STYLE_WEIGHT, description="スタイルの強さ"),
        reference_audio_path: Optional[str] = Query(None, description="スタイルを音声ファイルで指定"),
    ):
        """
        Voice Changer: 音声ファイルを別の話者の声に変換する (STT -> TTS)

        アップロードされた音声ファイルを Whisper で音声認識し、
        認識されたテキストを Style-Bert-VITS2 で音声合成します。
        """
        logger.info(
            f"{request.client.host}:{request.client.port}/voice_changer  "
            f"file={audio_file.filename}, model_id={model_id}, speaker_id={speaker_id}"
        )

        # Whisperモデルが初期化されているか確認
        if whisper_model is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Whisper model is not loaded. Voice changer is not available.",
            )

        # モデルIDのバリデーション
        if model_id >= len(model_holder.model_names):
            raise_validation_error(f"model_id={model_id} not found", "model_id")

        # モデル名が指定されている場合はmodel_idに変換
        if model_name:
            model_ids = [
                i
                for i, x in enumerate(model_holder.models_info)
                if x.name == model_name
            ]
            if not model_ids:
                raise_validation_error(f"model_name={model_name} not found", "model_name")
            if len(model_ids) > 1:
                raise_validation_error(f"model_name={model_name} is ambiguous", "model_name")
            model_id = model_ids[0]

        model = loaded_models[model_id]

        # 話者の検証
        if speaker_name is None:
            if speaker_id not in model.id2spk.keys():
                raise_validation_error(f"speaker_id={speaker_id} not found", "speaker_id")
        else:
            if speaker_name not in model.spk2id.keys():
                raise_validation_error(f"speaker_name={speaker_name} not found", "speaker_name")
            speaker_id = model.spk2id[speaker_name]

        # スタイルの検証
        if style not in model.style2id.keys():
            raise_validation_error(f"style={style} not found", "style")
        assert style is not None

        # 言語の変換
        language_map = {
            "ja": Languages.JP,
            "en": Languages.EN,
            "zh": Languages.ZH,
        }
        if language not in language_map:
            raise_validation_error(
                f"language={language} not supported. Use ja/en/zh", "language"
            )
        tts_language = language_map[language]

        # 一時ファイルに保存
        temp_audio_file = None
        try:
            # アップロードされたファイルを一時保存
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                temp_audio_file = temp_file.name
                content = await audio_file.read()
                temp_file.write(content)

            logger.info(f"Temporary audio file saved: {temp_audio_file}")

            # Whisper で音声認識
            logger.info("Starting speech recognition with Whisper...")
            initial_prompt = whisper_initial_prompt if whisper_initial_prompt else None
            transcribed_text = transcribe_with_faster_whisper(
                model=whisper_model,
                audio_file=Path(temp_audio_file),
                initial_prompt=initial_prompt,
                language=language,
                num_beams=1,
                no_repeat_ngram_size=10,
            )

            logger.info(f"Transcription completed: {transcribed_text}")

            if not transcribed_text or transcribed_text.strip() == "":
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="No speech detected in the audio file.",
                )

            # Style-Bert-VITS2 で音声合成
            logger.info("Starting TTS synthesis...")
            sr, audio = model.infer(
                text=transcribed_text,
                language=tts_language,
                speaker_id=speaker_id,
                reference_audio_path=reference_audio_path,
                sdp_ratio=sdp_ratio,
                noise=noise,
                noise_w=noisew,
                length=length,
                line_split=auto_split,
                split_interval=split_interval,
                assist_text=assist_text,
                assist_text_weight=assist_text_weight,
                use_assist_text=bool(assist_text),
                style=style,
                style_weight=style_weight,
            )

            logger.success(
                f"Voice conversion completed successfully. Transcribed text: {transcribed_text}"
            )

            # WAV ファイルとして返す
            with BytesIO() as wavContent:
                wavfile.write(wavContent, sr, audio)
                return Response(
                    content=wavContent.getvalue(),
                    media_type="audio/wav",
                    headers={
                        "X-Transcribed-Text": transcribed_text.encode("utf-8").decode("latin1", errors="ignore"),
                    },
                )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Voice changer error: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Voice conversion failed: {str(e)}",
            )
        finally:
            # 一時ファイルのクリーンアップ
            if temp_audio_file and os.path.exists(temp_audio_file):
                try:
                    os.unlink(temp_audio_file)
                    logger.info(f"Temporary file deleted: {temp_audio_file}")
                except Exception as e:
                    logger.warning(f"Failed to delete temporary file: {e}")

    @app.post("/g2p")
    def g2p(text: str):
        return g2kata_tone(normalize_text(text))

    @app.get("/models/info")
    def get_loaded_models_info():
        """ロードされたモデル情報の取得"""

        result: dict[str, dict[str, Any]] = dict()
        for model_id, model in enumerate(loaded_models):
            result[str(model_id)] = {
                "config_path": model.config_path,
                "model_path": model.model_path,
                "device": model.device,
                "spk2id": model.spk2id,
                "id2spk": model.id2spk,
                "style2id": model.style2id,
            }
        return result

    @app.post("/models/refresh")
    def refresh():
        """モデルをパスに追加/削除した際などに読み込ませる"""
        model_holder.refresh()
        load_models(model_holder)
        return get_loaded_models_info()

    @app.get("/status")
    def get_status():
        """実行環境のステータスを取得"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_info = psutil.virtual_memory()
        memory_total = memory_info.total
        memory_available = memory_info.available
        memory_used = memory_info.used
        memory_percent = memory_info.percent
        gpuInfo = []
        devices = ["cpu"]
        for i in range(torch.cuda.device_count()):
            devices.append(f"cuda:{i}")
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            gpuInfo.append(
                {
                    "gpu_id": gpu.id,
                    "gpu_load": gpu.load,
                    "gpu_memory": {
                        "total": gpu.memoryTotal,
                        "used": gpu.memoryUsed,
                        "free": gpu.memoryFree,
                    },
                }
            )
        return {
            "devices": devices,
            "cpu_percent": cpu_percent,
            "memory_total": memory_total,
            "memory_available": memory_available,
            "memory_used": memory_used,
            "memory_percent": memory_percent,
            "gpu": gpuInfo,
        }

    @app.get("/tools/get_audio", response_class=AudioResponse)
    def get_audio(
        request: Request, path: str = Query(..., description="local wav path")
    ):
        """wavデータを取得する"""
        logger.info(
            f"{request.client.host}:{request.client.port}/tools/get_audio  { unquote(str(request.query_params) )}"
        )
        if not os.path.isfile(path):
            raise_validation_error(f"path={path} not found", "path")
        if not path.lower().endswith(".wav"):
            raise_validation_error(f"wav file not found in {path}", "path")
        return FileResponse(path=path, media_type="audio/wav")

    logger.info(f"server listen: http://127.0.0.1:{config.server_config.port}")
    logger.info(f"API docs: http://127.0.0.1:{config.server_config.port}/docs")
    logger.info(
        f"Input text length limit: {limit}. You can change it in server.limit in config.yml"
    )
    uvicorn.run(
        app, port=config.server_config.port, host="0.0.0.0", log_level="warning"
    )
