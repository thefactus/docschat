'use client'

import { useState, useRef } from 'react'
import { createPortal } from 'react-dom'
import type { NodeState, TraceState } from '@/lib/sse-types'

// ── Node styling by state ─────────────────────────────────────────────────

const nodeStyles: Record<NodeState, string> = {
  idle: 'border-gray-200 bg-gray-50 text-gray-400',
  active: 'border-indigo-400 bg-indigo-50 text-indigo-700 animate-node-pulse',
  done: 'border-emerald-400 bg-emerald-50 text-emerald-700',
  failed: 'border-red-300 bg-red-50 text-red-700',
  refused: 'border-orange-400 bg-orange-50 text-orange-700',
  skipped: 'border-gray-100 bg-white text-gray-200 opacity-40',
}

const nodeIcons: Record<NodeState, string> = {
  idle: '○',
  active: '◉',
  done: '✓',
  failed: '✕',
  refused: '⊘',
  skipped: '—',
}

interface TooltipContent {
  description: string
  detail?: string
}

interface NodeProps {
  label: string
  state: NodeState
  detail?: string
  tooltip: TooltipContent
}

function DiagramNode({ label, state, detail, tooltip }: NodeProps) {
  const nodeRef = useRef<HTMLDivElement>(null)
  const [hovered, setHovered] = useState(false)
  const [coords, setCoords] = useState({ top: 0, left: 0 })

  const handleMouseEnter = () => {
    if (nodeRef.current) {
      const r = nodeRef.current.getBoundingClientRect()
      setCoords({ top: r.top - 10, left: r.left + r.width / 2 })
    }
    setHovered(true)
  }

  return (
    <div
      ref={nodeRef}
      className="flex-shrink-0"
      onMouseEnter={handleMouseEnter}
      onMouseLeave={() => setHovered(false)}
    >
      <div
        className={`flex flex-col items-center justify-center rounded-lg border px-3 py-2
                    min-w-[72px] text-center transition-all duration-300 cursor-default
                    ${nodeStyles[state]}`}
      >
        <div className="flex items-center gap-1">
          <span className="text-xs font-bold">{nodeIcons[state]}</span>
          <span className="text-xs font-semibold whitespace-nowrap">{label}</span>
        </div>
        {detail && (
          <span className="text-[10px] opacity-70 mt-0.5 whitespace-nowrap">{detail}</span>
        )}
      </div>

      {/* Portal tooltip — fixed above the node, outside any scrollable container */}
      {hovered &&
        typeof document !== 'undefined' &&
        createPortal(
          <div
            style={{
              position: 'fixed',
              top: coords.top,
              left: coords.left,
              transform: 'translate(-50%, -100%)',
              zIndex: 9999,
              pointerEvents: 'none',
            }}
            className="w-52"
          >
            <div className="bg-white border border-gray-200 rounded-lg shadow-xl px-3 py-2.5 text-left">
              <p className="text-[11px] text-gray-600 leading-relaxed">{tooltip.description}</p>
              {tooltip.detail && (
                <p className="text-[11px] font-mono text-indigo-600 mt-1.5 leading-relaxed border-t border-gray-100 pt-1.5">
                  {tooltip.detail}
                </p>
              )}
            </div>
            {/* Arrow pointing down */}
            <div
              className="absolute left-1/2 -translate-x-1/2 bottom-0 translate-y-full"
              style={{ width: 0, height: 0, borderLeft: '6px solid transparent', borderRight: '6px solid transparent', borderTop: '6px solid #e5e7eb' }}
            />
            <div
              className="absolute left-1/2 -translate-x-1/2 bottom-0 translate-y-[calc(100%-1px)]"
              style={{ width: 0, height: 0, borderLeft: '5px solid transparent', borderRight: '5px solid transparent', borderTop: '5px solid white' }}
            />
          </div>,
          document.body,
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

  // ── Node display details (sub-text) ──────────────────────────────────────
  const plannerDetail = trace.planner
    ? trace.planner.intent === 'greeting'
      ? 'greeting'
      : trace.planner.intent === 'meta'
      ? 'meta'
      : trace.planner.rewritten
      ? 'rewrote'
      : trace.planner.decomposed
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

  // ── Tooltip content ───────────────────────────────────────────────────────
  const plannerTooltip: TooltipContent = {
    description: trace.planner?.intent === 'greeting'
      ? 'Detected a greeting — short-circuit, no retrieval.'
      : trace.planner?.intent === 'meta'
      ? 'Detected a meta question — short-circuit.'
      : trace.planner?.rewritten
      ? 'Rewrote the follow-up into a standalone document question.'
      : trace.planner?.decomposed
      ? `Question split into ${trace.planner.sub_queries.length} sub-queries for broader retrieval coverage.`
      : trace.planner
      ? 'Question sent as-is — no decomposition needed.'
      : 'Decides whether to split your question into sub-queries for better retrieval.',
    detail: trace.planner?.rewritten
      ? trace.planner.standalone_query
      : trace.planner?.reason
      ? `Reason: ${trace.planner.reason}`
      : undefined,
  }

  const retrievalHits = trace.retrievals[0]
  const retrievalTooltip: TooltipContent = {
    description:
      'Hybrid search: semantic vector similarity + BM25 keyword search run in parallel across your documents.',
    detail: retrievalHits
      ? `${retrievalHits.vector_hits} semantic · ${retrievalHits.fts_hits} keyword → ${retrievalHits.fused} merged`
      : undefined,
  }

  const fusionTop = trace.fusion?.top?.slice(0, 2).map((c) => c.filename).join(', ')
  const fusionTooltip: TooltipContent = {
    description: `Merges both search arms with weighted scoring. ${
      trace.fusion?.weights ?? '0.7/0.3'
    } = semantic / keyword weight. Chunks that rank well in both arms are promoted.`,
    detail: fusionTop ? `Top: ${fusionTop}` : undefined,
  }

  const guardrailTooltip: TooltipContent = {
    description:
      'Confidence gate. If the best-matching chunk scores below the threshold, the answer is refused to avoid hallucinated responses.',
    detail: trace.guardrail
      ? `Score ${trace.guardrail.max_score.toFixed(2)} vs threshold ${
          trace.guardrail.threshold
        } → ${trace.guardrail.decision === 'refuse' ? 'refused ⊘' : 'passed ✓'}`
      : undefined,
  }

  const generationTooltip: TooltipContent = {
    description:
      'Top-ranked chunks are sent as context to the language model, which generates an answer grounded strictly in those passages.',
    detail:
      trace.chunks_used != null ? `${trace.chunks_used} chunks passed as context` : undefined,
  }

  return (
    <div className="flex items-center py-1 px-1 overflow-x-auto">
      <DiagramNode
        label="Query"
        state={nodes.query}
        tooltip={{ description: 'Your question, passed as input to every stage of the pipeline.' }}
      />
      <Edge flowing={nodes.planner === 'active'} />
      <DiagramNode
        label="Planner"
        state={nodes.planner}
        detail={plannerDetail}
        tooltip={plannerTooltip}
      />
      <Edge
        flowing={
          nodes.retrieval === 'active' ||
          (nodes.planner === 'done' && nodes.retrieval === 'idle')
        }
      />
      <DiagramNode
        label="Retrieval"
        state={nodes.retrieval}
        detail={retrievalDetail}
        tooltip={retrievalTooltip}
      />
      <Edge
        flowing={
          nodes.fusion === 'active' ||
          (nodes.retrieval === 'done' && nodes.fusion === 'idle')
        }
      />
      <DiagramNode
        label="Fusion"
        state={nodes.fusion}
        detail={fusionDetail}
        tooltip={fusionTooltip}
      />
      <Edge flowing={nodes.guardrail === 'active'} />
      <DiagramNode
        label="Guardrail"
        state={nodes.guardrail}
        detail={guardrailDetail}
        tooltip={guardrailTooltip}
      />
      <Edge flowing={nodes.generation === 'active'} />
      <DiagramNode
        label="LLM"
        state={nodes.generation}
        detail={generationDetail}
        tooltip={generationTooltip}
      />
      <Edge flowing={nodes.answer === 'active'} />
      <DiagramNode
        label="Answer"
        state={nodes.answer}
        tooltip={{
          description:
            'Final answer, grounded in your documents. Inline citations (e.g. [1]) link each claim back to its source chunk.',
        }}
      />
    </div>
  )
}
