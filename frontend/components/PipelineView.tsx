'use client'

import { useEffect, useRef } from 'react'
import { PipelineDiagram } from './PipelineDiagram'
import type { TraceState } from '@/lib/sse-types'

interface Props {
  trace: TraceState
  isStreaming: boolean
  collapsed: boolean
  onToggle: () => void
}

export function PipelineView({ trace, isStreaming, collapsed, onToggle }: Props) {
  const logRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight
    }
  }, [trace.log])

  // ── Collapsed chip ────────────────────────────────────────────────────
  if (collapsed && trace.done) {
    const { sources, latency_ms } = trace.done
    return (
      <button
        onClick={onToggle}
        className="inline-flex items-center gap-1.5 text-xs px-2.5 py-1 rounded-full
                   bg-indigo-50 border border-indigo-200 text-indigo-700
                   hover:bg-indigo-100 transition-colors mb-3"
      >
        <span className="font-mono text-[10px] text-indigo-400">◈</span>
        <span className="font-medium">Pipeline</span>
        <span className="text-indigo-400">·</span>
        <span>{sources.length} source{sources.length !== 1 ? 's' : ''}</span>
        <span className="text-indigo-400">·</span>
        <span>{(latency_ms / 1000).toFixed(1)}s</span>
        <svg className="w-3 h-3 ml-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
        </svg>
      </button>
    )
  }

  // ── Expanded view ──────────────────────────────────────────────────────
  return (
    <div className="rounded-xl border border-indigo-100 bg-indigo-50/40 mb-3 overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-2 border-b border-indigo-100">
        <div className="flex items-center gap-2">
          <span className="text-[10px] font-mono text-indigo-400">◈</span>
          <span className="text-xs font-semibold text-indigo-700">Pipeline trace</span>
          {isStreaming && (
            <span className="flex items-center gap-1 text-[10px] text-indigo-500">
              <span className="w-1.5 h-1.5 rounded-full bg-indigo-400 animate-pulse" />
              live
            </span>
          )}
        </div>
        {trace.done && (
          <button
            onClick={onToggle}
            className="text-[10px] text-indigo-400 hover:text-indigo-600 transition-colors"
          >
            collapse ×
          </button>
        )}
      </div>

      {/* Diagram */}
      <div className="px-3 pt-3 pb-1">
        <PipelineDiagram trace={trace} />
      </div>

      {/* Log */}
      {trace.log.length > 0 && (
        <div
          ref={logRef}
          className="mx-3 mb-3 rounded-lg bg-gray-900 px-3 py-2 max-h-28 overflow-y-auto"
        >
          {trace.log.map((line, i) => (
            <div
              key={i}
              className="text-[11px] font-mono text-gray-300 leading-relaxed animate-fade-in"
            >
              <span className="text-gray-600 select-none mr-2">›</span>
              {line}
            </div>
          ))}
          {isStreaming && (
            <span className="text-[11px] font-mono text-gray-500 animate-cursor">▊</span>
          )}
        </div>
      )}

      {/* Error state */}
      {trace.error && (
        <div className="mx-3 mb-3 rounded-lg bg-red-50 border border-red-200 px-3 py-2">
          <p className="text-xs text-red-700 font-mono">{trace.error}</p>
        </div>
      )}
    </div>
  )
}
