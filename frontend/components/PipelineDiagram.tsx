'use client'

import type { NodeState, TraceState } from '@/lib/sse-types'

// ── Node styling by state ─────────────────────────────────────────────────

const nodeStyles: Record<NodeState, string> = {
  idle: 'border-gray-200 bg-gray-50 text-gray-400',
  active: 'border-indigo-400 bg-indigo-50 text-indigo-700 animate-node-pulse',
  done: 'border-emerald-400 bg-emerald-50 text-emerald-700',
  failed: 'border-red-300 bg-red-50 text-red-700',
  refused: 'border-orange-400 bg-orange-50 text-orange-700',
}

const nodeIcons: Record<NodeState, string> = {
  idle: '○',
  active: '◉',
  done: '✓',
  failed: '✕',
  refused: '⊘',
}

interface NodeProps {
  label: string
  state: NodeState
  detail?: string
}

function DiagramNode({ label, state, detail }: NodeProps) {
  return (
    <div
      className={`flex flex-col items-center justify-center rounded-lg border px-3 py-2
                  min-w-[72px] text-center transition-all duration-300 ${nodeStyles[state]}`}
    >
      <div className="flex items-center gap-1">
        <span className="text-xs font-bold">{nodeIcons[state]}</span>
        <span className="text-xs font-semibold whitespace-nowrap">{label}</span>
      </div>
      {detail && (
        <span className="text-[10px] opacity-70 mt-0.5 whitespace-nowrap">{detail}</span>
      )}
    </div>
  )
}

// ── Edge (connector) ──────────────────────────────────────────────────────

function Edge({ flowing }: { flowing: boolean }) {
  return (
    <div className="relative flex-shrink-0 h-0.5 w-6 bg-gray-200 self-center mx-0.5 overflow-hidden rounded-full">
      {flowing && (
        <div className="absolute top-0 bottom-0 w-3 rounded-full bg-indigo-400 animate-flow-dot" />
      )}
    </div>
  )
}

// ── Main diagram ──────────────────────────────────────────────────────────

export function PipelineDiagram({ trace }: { trace: TraceState }) {
  const { nodes } = trace

  const plannerDetail = trace.planner
    ? trace.planner.decomposed
      ? `${trace.planner.sub_queries.length} sub-q`
      : '1 query'
    : undefined

  const retrievalDetail =
    trace.retrievals.length > 0
      ? `×${trace.retrievals.length} arm${trace.retrievals.length > 1 ? 's' : ''}`
      : undefined

  const fusionDetail = trace.fusion?.weights

  const guardrailDetail = trace.guardrail
    ? `${trace.guardrail.max_score.toFixed(2)} / ${trace.guardrail.threshold}`
    : undefined

  const generationDetail =
    trace.chunks_used != null ? `${trace.chunks_used} chunks` : undefined

  return (
    <div className="flex items-center py-1 px-1 overflow-x-auto">
      <DiagramNode label="Query" state={nodes.query} />
      <Edge flowing={nodes.planner === 'active'} />
      <DiagramNode label="Planner" state={nodes.planner} detail={plannerDetail} />
      <Edge flowing={nodes.retrieval === 'active' || (nodes.planner === 'done' && nodes.retrieval === 'idle')} />
      <DiagramNode label="Retrieval" state={nodes.retrieval} detail={retrievalDetail} />
      <Edge flowing={nodes.fusion === 'active' || (nodes.retrieval === 'done' && nodes.fusion === 'idle')} />
      <DiagramNode label="Fusion" state={nodes.fusion} detail={fusionDetail} />
      <Edge flowing={nodes.guardrail === 'active'} />
      <DiagramNode label="Guardrail" state={nodes.guardrail} detail={guardrailDetail} />
      <Edge flowing={nodes.generation === 'active'} />
      <DiagramNode label="LLM" state={nodes.generation} detail={generationDetail} />
      <Edge flowing={nodes.answer === 'active'} />
      <DiagramNode label="Answer" state={nodes.answer} />
    </div>
  )
}
