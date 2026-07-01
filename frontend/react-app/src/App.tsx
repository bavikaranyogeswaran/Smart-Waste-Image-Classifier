import { useState } from 'react'
import axios from 'axios'

interface PredictionResult {
  class: string
  confidence: number
}

function App() {
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [result, setResult] = useState<PredictionResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selected = e.target.files?.[0] ?? null
    setFile(selected)
    setResult(null)
    setError(null)
    if (selected) {
      setPreview(URL.createObjectURL(selected))
    }
  }

  const handleUpload = async () => {
    if (!file) return
    setLoading(true)
    setError(null)
    try {
      const formData = new FormData()
      formData.append('file', file)
      const response = await axios.post<PredictionResult>(
        'http://127.0.0.1:8000/predict',
        formData,
        { headers: { 'Content-Type': 'multipart/form-data' } }
      )
      setResult(response.data)
    } catch {
      setError('Prediction failed. Make sure the backend is running.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={{ maxWidth: 600, margin: '60px auto', padding: '0 24px', fontFamily: 'Arial, sans-serif' }}>
      <h1>Smart Waste Classifier</h1>

      <input type="file" accept="image/*" onChange={handleFileChange} />

      {preview && (
        <div style={{ marginTop: 16 }}>
          <h3>Preview</h3>
          <img src={preview} alt="preview" style={{ width: 300, borderRadius: 8 }} />
        </div>
      )}

      <div style={{ marginTop: 16 }}>
        <button onClick={handleUpload} disabled={!file || loading}>
          {loading ? 'Predicting...' : 'Predict'}
        </button>
      </div>

      {error && <p style={{ color: 'red', marginTop: 16 }}>{error}</p>}

      {result && (
        <div style={{ marginTop: 24, padding: 16, border: '1px solid #ccc', borderRadius: 8 }}>
          <h2>Prediction Result</h2>
          <p><strong>Class:</strong> {result.class}</p>
          <p><strong>Confidence:</strong> {(result.confidence * 100).toFixed(2)}%</p>
        </div>
      )}
    </div>
  )
}

export default App
