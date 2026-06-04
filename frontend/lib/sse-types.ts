import type { Source } from './types'

// ── SSE event payloads ────────────────────────────────────────────────────

export interface PlannerData {
  decomposed: boolean
  sub_queries: string[]
  reason: string
}

export interface RetrievalData {
  sub_query: string
  vector_hits: number
  fts_hits: number
  fused: number
}

export interface FusionData {
  weights: string
  top: Array<{ filename: string; page: number | null; score: number }>
}

export interface GuardrailData {
  max_score: number
  threshold: number
  decision: 'proceed' | 'refuse'
}

export interface GeneratingData {
  chunks_used: number
}

export interface DoneData {
  answer: string
  sources: Source[]
  tokens_used: number
  latency_ms: number
}

export type ParsedSSEEvent =
  | { event: 'planner'; data: PlannerData }
  | { event: 'retrieval'; data: RetrievalData }
  | { event: 'fusion'; data: FusionData }
  | { event: 'guardrail'; data: GuardrailData }
  | { event: 'generating'; data: GeneratingData }
  | { event: 'token'; data: { text: string } }
  | { event: 'done'; data: DoneData }
  | { event: 'error'; data: { message: string } }

// ── Pipeline node model ───────────────────────────────────────────────────

export type PipelineNodeId =
  | 'query'
  | 'planner'
  | 'retrieval'
  | 'fusion'
  | 'guardrail'
  | 'generation'
  | 'answer'

export type NodeState = 'idle' | 'active' | 'done' | 'failed' | 'refused'

// ── Trace state ────────────────────────────────────────────────────────────

export interface TraceState {
  nodes: Record<PipelineNodeId, NodeState>
  planner: PlannerData | null
  retrievals: RetrievalData[]
  fusion: FusionData | null
  guardrail: GuardrailData | null
  chunks_used: number | null
  answer: string
  done: DoneData | null
  error: string | null
  log: string[]
}

export function initialTraceState(): TraceState {
  return {
    nodes: {
      query: 'done',
      planner: 'active',
      retrieval: 'idle',
      fusion: 'idle',
      guardrail: 'idle',
      generation: 'idle',
      answer: 'idle',
    },
    planner: null,
    retrievals: [],
    fusion: null,
    guardrail: null,
    chunks_used: null,
    answer: '',
    done: null,
    error: null,
    log: [],
  }
}

// ── Trace reducer ──────────────────────────────────────────────────────────

export function reduceTraceEvent(state: TraceState, event: ParsedSSEEvent): TraceState {
  switch (event.event) {
    case 'planner': {
      const d = event.data
      const n = d.sub_queries.length
      const logLine = d.decomposed
        ? `Planner → ${n} sub-queries (decomposed)`
        : `Planner → single query`
      return {
        ...state,
        nodes: { ...state.nodes, planner: 'done', retrieval: 'active' },
        planner: d,
        log: [...state.log, logLine],
      }
    }

    case 'retrieval': {
      const d = event.data
      const logLine = `Hybrid retrieval → vector ${d.vector_hits} · FTS ${d.fts_hits} → fused to ${d.fused}`
      return {
        ...state,
        retrievals: [...state.retrievals, d],
        log: [...state.log, logLine],
      }
    }

    case 'fusion': {
      const d = event.data
      const logLine = `Fusion (${d.weights}) → top ${d.top.length} chunks`
      return {
        ...state,
        nodes: { ...state.nodes, retrieval: 'done', fusion: 'done', guardrail: 'active' },
        fusion: d,
        log: [...state.log, logLine],
      }
    }

    case 'guardrail': {
      const d = event.data
      const sign = d.decision === 'proceed' ? '>=' : '<'
      const logLine = `Confidence → max ${d.max_score.toFixed(2)} ${sign} ${d.threshold} → ${d.decision}`
      const guardrailState: NodeState = d.decision === 'refuse' ? 'refused' : 'done'
      return {
        ...state,
        nodes: {
          ...state.nodes,
          guardrail: guardrailState,
          generation: d.decision === 'proceed' ? 'active' : 'idle',
        },
        guardrail: d,
        log: [...state.log, logLine],
      }
    }

    case 'generating': {
      const d = event.data
      return {
        ...state,
        chunks_used: d.chunks_used,
        log: [...state.log, `Generating from ${d.chunks_used} chunks...`],
      }
    }

    case 'token': {
      return {
        ...state,
        nodes: { ...state.nodes, answer: 'active' },
        answer: state.answer + event.data.text,
      }
    }

    case 'done': {
      const d = event.data
      const logLine = `Done in ${(d.latency_ms / 1000).toFixed(1)}s`
      return {
        ...state,
        nodes: { ...state.nodes, generation: 'done', answer: 'done' },
        done: d,
        log: [...state.log, logLine],
      }
    }

    case 'error': {
      return {
        ...state,
        error: event.data.message,
        log: [...state.log, `Error: ${event.data.message}`],
      }
    }

    default:
      return state
  }
}
