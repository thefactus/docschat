/**
 * Two interchangeable trace drivers.
 *
 * Driver interface:
 *   (question, documentIds, onEvent, onError) => { abort }
 *
 * SSE driver    — real POST /query/stream, fetch + ReadableStream.
 * Simulated driver — calls POST /query, then replays fake stage events
 *                    so the diagram always works even if SSE is unavailable.
 *
 * Swap drivers by changing the `activeDriver` export below.
 */

import type { ParsedSSEEvent } from './sse-types'
import type { QueryResponse } from './api'

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000'

export type TraceDriver = (
  question: string,
  documentIds: string[] | null,
  onEvent: (event: ParsedSSEEvent) => void,
  onError: (msg: string) => void,
) => { abort: () => void }

// ── SSE buffer parser (pure — used by sseDriver and testable standalone) ──

export function parseSSEBuffer(buffer: string): {
  parsed: ParsedSSEEvent[]
  remaining: string
} {
  const parsed: ParsedSSEEvent[] = []
  const parts = buffer.split('\n\n')
  const remaining = parts.pop() ?? ''

  for (const part of parts) {
    if (!part.trim()) continue
    let eventName = ''
    let dataLine = ''

    for (const line of part.split('\n')) {
      if (line.startsWith('event: ')) eventName = line.slice(7).trim()
      else if (line.startsWith('data: ')) dataLine = line.slice(6)
    }

    if (eventName && dataLine) {
      try {
        parsed.push({ event: eventName, data: JSON.parse(dataLine) } as ParsedSSEEvent)
      } catch {
        // skip malformed event
      }
    }
  }

  return { parsed, remaining }
}

// ── SSE driver ────────────────────────────────────────────────────────────

export const sseDriver: TraceDriver = (question, documentIds, onEvent, onError) => {
  const controller = new AbortController()

  const run = async () => {
    const res = await fetch(`${API_URL}/query/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        question,
        document_ids: documentIds?.length ? documentIds : null,
      }),
      signal: controller.signal,
    })

    if (!res.ok || !res.body) {
      throw new Error(`HTTP ${res.status}`)
    }

    const reader = res.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''

    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      buffer += decoder.decode(value, { stream: true })
      const result = parseSSEBuffer(buffer)
      buffer = result.remaining

      for (const evt of result.parsed) {
        onEvent(evt)
      }
    }
  }

  run().catch((err) => {
    if (err.name !== 'AbortError') {
      onError((err as Error).message ?? 'Stream error')
    }
  })

  return { abort: () => controller.abort() }
}

// ── Simulated driver ──────────────────────────────────────────────────────
// Calls /query (the verified path) to get authentic answer + sources,
// then replays fake stage events with realistic timing so the diagram works.

export const simulatedDriver: TraceDriver = (question, documentIds, onEvent, onError) => {
  let aborted = false
  const timers: ReturnType<typeof setTimeout>[] = []

  const schedule = (fn: () => void, delay: number) => {
    if (aborted) return
    timers.push(setTimeout(() => { if (!aborted) fn() }, delay))
  }

  const run = async () => {
    const res = await fetch(`${API_URL}/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        question,
        document_ids: documentIds?.length ? documentIds : null,
      }),
    })

    if (!res.ok) throw new Error(`HTTP ${res.status}`)
    const qr = (await res.json()) as QueryResponse
    if (aborted) return

    let t = 0
    const refused = qr.tokens_used === 0
    const maxScore = qr.sources.length > 0 ? Math.max(...qr.sources.map((s) => s.score)) : 0.15

    // Planner
    schedule(() => onEvent({
      event: 'planner',
      data: { decomposed: false, sub_queries: [question], reason: 'simulated' },
    }), t)
    t += 320

    // Retrieval
    const fusedCount = qr.sources.length || 6
    schedule(() => onEvent({
      event: 'retrieval',
      data: { sub_query: question, vector_hits: 15, fts_hits: 8, fused: fusedCount },
    }), t)
    t += 380

    // Fusion
    schedule(() => onEvent({
      event: 'fusion',
      data: {
        weights: '0.7/0.3',
        top: qr.sources
          .slice(0, 5)
          .map((s) => ({ filename: s.filename, page: s.page, score: s.score })),
      },
    }), t)
    t += 200

    // Guardrail
    const decision = refused ? 'refuse' : 'proceed'
    schedule(() => onEvent({
      event: 'guardrail',
      data: { max_score: maxScore, threshold: 0.2, decision },
    }), t)
    t += 180

    if (refused) {
      schedule(() => onEvent({
        event: 'done',
        data: {
          answer: qr.answer,
          sources: [],
          tokens_used: 0,
          latency_ms: qr.latency_ms,
        },
      }), t)
      return
    }

    // Generating
    schedule(() => onEvent({
      event: 'generating',
      data: { chunks_used: fusedCount },
    }), t)
    t += 120

    // Token streaming — split answer into small pieces for a typing effect
    const pieces = qr.answer.match(/[\s\S]{1,4}/g) ?? []
    pieces.forEach((piece, i) => {
      schedule(() => onEvent({ event: 'token', data: { text: piece } }), t + i * 25)
    })
    t += pieces.length * 25 + 120

    // Done
    schedule(() => onEvent({
      event: 'done',
      data: {
        answer: qr.answer,
        sources: qr.sources,
        tokens_used: qr.tokens_used,
        latency_ms: qr.latency_ms,
      },
    }), t)
  }

  run().catch((err) => {
    if (!aborted) onError((err as Error).message ?? 'Simulated stream error')
  })

  return {
    abort: () => {
      aborted = true
      timers.forEach(clearTimeout)
    },
  }
}

// ── Active driver (flip this to simulatedDriver if SSE is unavailable) ───
export const activeDriver: TraceDriver = sseDriver
