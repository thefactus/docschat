import type { Document, Source } from './types'

const API_URL =
  process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000'

export interface IngestResponse {
  document_id: string
  filename: string
  chunks: number
  message: string
}

export interface QueryResponse {
  answer: string
  sources: Source[]
  tokens_used: number
  latency_ms: number
  intent?: string
}

export interface HistoryMessage {
  role: 'user' | 'assistant'
  content: string
}

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.json().catch(() => null)
    const detail = body?.detail ?? `HTTP ${res.status}`
    throw new Error(String(detail))
  }
  return res.json() as Promise<T>
}

export async function uploadDocument(file: File): Promise<IngestResponse> {
  const form = new FormData()
  form.append('file', file)
  const res = await fetch(`${API_URL}/ingest`, { method: 'POST', body: form })
  return handleResponse<IngestResponse>(res)
}

export async function listDocuments(): Promise<Document[]> {
  const res = await fetch(`${API_URL}/documents`)
  const data = await handleResponse<{ documents: Document[] }>(res)
  return data.documents
}

export async function sendQuery(
  question: string,
  documentIds: string[] | null,
  history?: HistoryMessage[],
): Promise<QueryResponse> {
  const res = await fetch(`${API_URL}/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      question,
      document_ids: documentIds?.length ? documentIds : null,
      history: history ?? [],
    }),
  })
  return handleResponse<QueryResponse>(res)
}
