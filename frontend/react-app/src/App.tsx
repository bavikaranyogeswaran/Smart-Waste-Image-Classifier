import { useState } from 'react'
import axios from 'axios'

interface PredictionResult {
  class: string
  confidence: number
}

const CLASS_COLORS: Record<string, string> = {
  plastic:  'bg-blue-100 text-blue-800 border-blue-300',
  paper:    'bg-yellow-100 text-yellow-800 border-yellow-300',
  metal:    'bg-gray-100 text-gray-800 border-gray-300',
  glass:    'bg-cyan-100 text-cyan-800 border-cyan-300',
  organic:  'bg-green-100 text-green-800 border-green-300',
  e_waste:  'bg-red-100 text-red-800 border-red-300',
}

const CLASS_ICONS: Record<string, string> = {
  plastic: '🧴',
  paper:   '📄',
  metal:   '🥫',
  glass:   '🍶',
  organic: '🌿',
  e_waste: '🔋',
}

function ConfidenceBar({ value }: { value: number }) {
  const pct = Math.round(value * 100)
  const color = pct >= 80 ? 'bg-green-500' : pct >= 60 ? 'bg-yellow-500' : 'bg-red-500'
  return (
    <div className="mt-2">
      <div className="flex justify-between text-sm text-gray-500 mb-1">
        <span>Confidence</span>
        <span className="font-semibold">{pct}%</span>
      </div>
      <div className="w-full bg-gray-200 rounded-full h-3">
        <div
          className={`${color} h-3 rounded-full transition-all duration-700`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  )
}

function App() {
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [result, setResult] = useState<PredictionResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [dragOver, setDragOver] = useState(false)

  const handleFile = (selected: File | null) => {
    if (!selected) return
    setFile(selected)
    setResult(null)
    setError(null)
    setPreview(URL.createObjectURL(selected))
  }

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    handleFile(e.target.files?.[0] ?? null)
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setDragOver(false)
    handleFile(e.dataTransfer.files?.[0] ?? null)
  }

  const handleUpload = async () => {
    if (!file) return
    setLoading(true)
    setError(null)
    try {
      const formData = new FormData()
      formData.append('file', file)
      const response = await axios.post<PredictionResult>(
        `${import.meta.env.VITE_API_URL ?? 'http://127.0.0.1:8000'}/predict`,
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

  const colorClass = result ? (CLASS_COLORS[result.class] ?? 'bg-gray-100 text-gray-800 border-gray-300') : ''
  const icon = result ? (CLASS_ICONS[result.class] ?? '♻️') : ''

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 flex items-center justify-center p-6">
      <div className="w-full max-w-lg">

        {/* Header */}
        <div className="text-center mb-8">
          <div className="text-5xl mb-3">♻️</div>
          <h1 className="text-3xl font-bold text-slate-800">Smart Waste Classifier</h1>
          <p className="text-slate-500 mt-2 text-sm">Upload a waste image to classify it instantly</p>
        </div>

        {/* Card */}
        <div className="bg-white rounded-2xl shadow-lg p-6 space-y-5">

          {/* Drop zone */}
          <div
            onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
            onDragLeave={() => setDragOver(false)}
            onDrop={handleDrop}
            className={`border-2 border-dashed rounded-xl p-6 text-center cursor-pointer transition-colors
              ${dragOver ? 'border-indigo-400 bg-indigo-50' : 'border-slate-300 hover:border-indigo-400 hover:bg-slate-50'}`}
          >
            <label className="cursor-pointer block">
              <div className="text-3xl mb-2">📁</div>
              <p className="text-slate-600 text-sm font-medium">
                {file ? file.name : 'Drop an image here or click to browse'}
              </p>
              <p className="text-slate-400 text-xs mt-1">JPG, PNG, WebP supported</p>
              <input
                type="file"
                accept="image/*"
                onChange={handleFileChange}
                className="hidden"
              />
            </label>
          </div>

          {/* Preview */}
          {preview && (
            <div className="rounded-xl overflow-hidden border border-slate-200">
              <img src={preview} alt="preview" className="w-full object-cover max-h-64" />
            </div>
          )}

          {/* Predict button */}
          <button
            onClick={handleUpload}
            disabled={!file || loading}
            className="w-full py-3 px-4 rounded-xl font-semibold text-white text-sm
              bg-indigo-600 hover:bg-indigo-700 active:bg-indigo-800
              disabled:bg-slate-300 disabled:cursor-not-allowed
              transition-colors duration-200"
          >
            {loading ? (
              <span className="flex items-center justify-center gap-2">
                <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24" fill="none">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z"/>
                </svg>
                Classifying...
              </span>
            ) : 'Classify Waste'}
          </button>

          {/* Error */}
          {error && (
            <div className="bg-red-50 border border-red-200 text-red-700 text-sm rounded-xl px-4 py-3">
              {error}
            </div>
          )}

          {/* Result */}
          {result && (
            <div className={`border rounded-xl px-5 py-4 ${colorClass}`}>
              <div className="flex items-center gap-3 mb-3">
                <span className="text-3xl">{icon}</span>
                <div>
                  <p className="text-xs font-medium uppercase tracking-widest opacity-60">Detected class</p>
                  <p className="text-xl font-bold capitalize">{result.class.replace('_', ' ')}</p>
                </div>
              </div>
              <ConfidenceBar value={result.confidence} />
            </div>
          )}
        </div>

        {/* Classes legend */}
        <div className="mt-6 flex flex-wrap justify-center gap-2">
          {Object.entries(CLASS_ICONS).map(([cls, icon]) => (
            <span key={cls} className={`text-xs px-3 py-1 rounded-full border font-medium ${CLASS_COLORS[cls]}`}>
              {icon} {cls.replace('_', ' ')}
            </span>
          ))}
        </div>

      </div>
    </div>
  )
}

export default App
