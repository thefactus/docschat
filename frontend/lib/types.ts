export interface Document {
  document_id: string
  filename: string
  chunks: number
}

export interface Source {
  document_id: string
  filename: string
  page: number | null
  chunk_index: number
  score: number
}

export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  sources?: Source[]
  tokensUsed?: number
  latencyMs?: number
  error?: boolean
}

export type UploadStatus =
  | { kind: 'idle' }
  | { kind: 'uploading'; filename: string }
  | { kind: 'indexing'; filename: string }
  | { kind: 'done'; filename: string; chunks: number }
  | { kind: 'error'; message: string }
