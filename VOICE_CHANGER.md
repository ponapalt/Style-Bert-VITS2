# Voice Changer 機能ドキュメント

## 概要

Voice Changer 機能は、アップロードされた音声ファイルを別の話者の声に変換する機能です。

**処理フロー:**
```
入力音声ファイル (WAV等)
    ↓
[Whisper による音声認識 (STT)]
    ↓
認識されたテキスト
    ↓
[Style-Bert-VITS2 による音声合成 (TTS)]
    ↓
変換後の音声ファイル (WAV)
```

## 必要な環境

- faster-whisper がインストールされていること
- Style-Bert-VITS2 のモデルが `model_assets/` に配置されていること
- CUDA 環境推奨（CPU でも動作可能だが遅い）

## API エンドポイント

### POST /voice_changer

音声ファイルを別の話者の声に変換します。

#### リクエスト

**Content-Type:** `multipart/form-data`

**パラメータ:**

| パラメータ | 型 | 必須 | デフォルト | 説明 |
|-----------|-----|------|-----------|------|
| `audio_file` | File | ✅ | - | 変換元の音声ファイル (WAV形式推奨) |
| `model_name` | string | | None | 変換先モデル名 (model_id より優先) |
| `model_id` | int | | 0 | 変換先モデルID |
| `speaker_name` | string | | None | 変換先話者名 (speaker_id より優先) |
| `speaker_id` | int | | 0 | 変換先話者ID |
| `language` | string | | "ja" | 音声認識言語 (ja/en/zh) |
| `whisper_initial_prompt` | string | | "" | Whisper 認識の初期プロンプト |
| `sdp_ratio` | float | | 0.2 | SDP/DP混合比 |
| `noise` | float | | 0.6 | サンプルノイズの割合 |
| `noisew` | float | | 0.8 | SDPノイズ |
| `length` | float | | 1.0 | 話速 (大きいほど遅い) |
| `auto_split` | bool | | true | 改行で分けて生成 |
| `split_interval` | float | | 0.5 | 分割時の無音の長さ（秒） |
| `assist_text` | string | | None | 感情表現の参照テキスト |
| `assist_text_weight` | float | | 0.7 | assist_text の強さ |
| `style` | string | | "Neutral" | スタイル |
| `style_weight` | float | | 1.0 | スタイルの強さ |
| `reference_audio_path` | string | | None | スタイルを音声ファイルで指定 |

#### レスポンス

**Content-Type:** `audio/wav`

変換後の音声データ (WAV 形式)

**カスタムヘッダー:**
- `X-Transcribed-Text`: 認識されたテキスト (UTF-8)

#### エラーレスポンス

- `400 Bad Request`: 音声が認識できなかった場合
- `422 Unprocessable Entity`: パラメータが不正な場合
- `500 Internal Server Error`: サーバー内部エラー
- `503 Service Unavailable`: Whisper モデルが読み込まれていない場合

## 使用例

### cURL を使用した例

```bash
# 基本的な使用例
curl -X POST "http://127.0.0.1:5000/voice_changer" \
  -F "audio_file=@input.wav" \
  -F "model_id=0" \
  -F "speaker_id=0" \
  -F "language=ja" \
  -o output.wav

# モデル名と話者名を指定
curl -X POST "http://127.0.0.1:5000/voice_changer" \
  -F "audio_file=@input.wav" \
  -F "model_name=my_model" \
  -F "speaker_name=Speaker1" \
  -F "style=Happy" \
  -o output.wav

# 詳細なパラメータ指定
curl -X POST "http://127.0.0.1:5000/voice_changer" \
  -F "audio_file=@input.wav" \
  -F "model_id=0" \
  -F "speaker_id=0" \
  -F "language=ja" \
  -F "whisper_initial_prompt=こんにちは。元気ですか？" \
  -F "sdp_ratio=0.3" \
  -F "noise=0.5" \
  -F "length=1.0" \
  -F "style=Neutral" \
  -F "style_weight=1.0" \
  -o output.wav
```

### Python (requests) を使用した例

```python
import requests

# 音声ファイルをアップロード
with open("input.wav", "rb") as f:
    files = {"audio_file": f}
    data = {
        "model_id": 0,
        "speaker_id": 0,
        "language": "ja",
        "style": "Neutral",
    }

    response = requests.post(
        "http://127.0.0.1:5000/voice_changer",
        files=files,
        data=data,
    )

    # レスポンスの確認
    if response.status_code == 200:
        # 音声ファイルを保存
        with open("output.wav", "wb") as out:
            out.write(response.content)

        # 認識されたテキストを取得
        transcribed_text = response.headers.get("X-Transcribed-Text", "")
        print(f"認識されたテキスト: {transcribed_text}")
    else:
        print(f"エラー: {response.status_code}")
        print(response.json())
```

### JavaScript (Fetch API) を使用した例

```javascript
const formData = new FormData();
formData.append('audio_file', audioFileInput.files[0]);
formData.append('model_id', '0');
formData.append('speaker_id', '0');
formData.append('language', 'ja');
formData.append('style', 'Neutral');

fetch('http://127.0.0.1:5000/voice_changer', {
  method: 'POST',
  body: formData
})
.then(response => {
  if (response.ok) {
    const transcribedText = response.headers.get('X-Transcribed-Text');
    console.log('認識されたテキスト:', transcribedText);
    return response.blob();
  }
  throw new Error('Voice conversion failed');
})
.then(blob => {
  // 音声をダウンロード
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'output.wav';
  a.click();
})
.catch(error => console.error('Error:', error));
```

## 利用可能なモデルとスピーカーの確認

### GET /models/info

利用可能なモデルと話者の一覧を取得します。

```bash
curl http://127.0.0.1:5000/models/info
```

レスポンス例:
```json
{
  "0": {
    "config_path": "model_assets/my_model/config.json",
    "model_path": "model_assets/my_model/model.safetensors",
    "device": "cuda",
    "spk2id": {
      "Speaker1": 0,
      "Speaker2": 1
    },
    "id2spk": {
      "0": "Speaker1",
      "1": "Speaker2"
    },
    "style2id": {
      "Neutral": 0,
      "Happy": 1,
      "Sad": 2
    }
  }
}
```

## パフォーマンス

### 処理時間の目安 (CUDA 使用時)

- **10秒の音声**: 約3-5秒
  - 音声認識: 約2-3秒
  - 音声合成: 約1-2秒

### メモリ使用量

- **Whisper (large-v3)**: 約3GB VRAM
- **TTS モデル**: 約1-2GB VRAM
- **合計**: 約4-5GB VRAM

### CPU モードでの動作

CUDA が利用できない環境でも動作しますが、処理時間が大幅に増加します。

- **10秒の音声**: 約30-60秒程度

## トラブルシューティング

### Whisper モデルが読み込まれない

**エラー:** `503 Service Unavailable: Whisper model is not loaded`

**解決策:**
1. `faster-whisper` がインストールされているか確認
   ```bash
   pip install faster-whisper
   ```

2. サーバー起動時のログを確認
   ```
   Loading Whisper model for voice changer...
   Whisper model loaded successfully
   ```

### 音声が認識されない

**エラー:** `400 Bad Request: No speech detected in the audio file`

**原因:**
- 音声ファイルに音声が含まれていない
- ノイズが多すぎる
- 音量が小さすぎる

**解決策:**
1. 音声ファイルを確認
2. `whisper_initial_prompt` を設定して認識精度を向上
3. 音量を調整

### メモリ不足エラー

**エラー:** CUDA out of memory

**解決策:**
1. CPU モードで起動
   ```bash
   python server_fastapi.py --cpu
   ```

2. 他のプロセスを終了して VRAM を確保

## サーバーの起動

```bash
# CUDA 使用 (推奨)
python server_fastapi.py

# CPU 使用
python server_fastapi.py --cpu

# カスタムモデルディレクトリ指定
python server_fastapi.py -d /path/to/models
```

起動後、以下の URL でアクセス可能:
- API サーバー: http://127.0.0.1:5000
- API ドキュメント: http://127.0.0.1:5000/docs

## 注意事項

1. **一時ファイル**: アップロードされた音声は一時ファイルとして保存され、処理後自動的に削除されます

2. **ファイルサイズ制限**: 大きな音声ファイルはアップロードに時間がかかります。長い音声は事前に分割することを推奨します

3. **同時リクエスト**: Whisper モデルは1つだけロードされているため、複数の同時リクエストは順次処理されます

4. **言語の一貫性**: 音声認識の `language` パラメータと TTS の言語は自動的に一致するように設定されます

## ライセンス

このコードは Style-Bert-VITS2 プロジェクトの一部として、同じライセンスの下で提供されます。
