'use client'

import { useState } from 'react'
import { PipelineView } from './PipelineView'
import type { Message } from '@/lib/types'

export function ChatMessage({ message }: { message: Message }) {
  const [sourcesOpen, setSourcesOpen] = useState(false)
  const [pipelineCollapsed, setPipelineCollapsed] = useState(false)
  const isUser = message.role === 'user'
  const hasSources = message.sources && message.sources.length > 0

  // ── Pipeline-traced assistant message ─────────────────────────────────
  if (!isUser && message.trace) {
    const isStreaming = !!message.streaming
    const hasAnswer = message.trace.answer.length > 0 || !!message.trace.done

    return (
      <div className="flex justify-start">
        <div className="max-w-[90%] flex flex-col gap-1">
          {/* Pipeline view (diagram + log), collapses to chip when done */}
          <PipelineView
            trace={message.trace}
            isStreaming={isStreaming}
            collapsed={pipelineCollapsed && !!message.trace.done}
            onToggle={() => setPipelineCollapsed((c) => !c)}
          />

          {/* Answer types in below the diagram */}
          {hasAnswer && (
            <div
              className="rounded-2xl rounded-bl-sm px-4 py-3 text-sm leading-relaxed
                         bg-white text-gray-800 border border-gray-200 shadow-sm"
            >
              <p className="whitespace-pre-wrap">
                {message.trace.answer || message.content}
                {isStreaming && message.trace.answer && (
                  <span className="inline-block w-0.5 h-3.5 bg-gray-500 ml-0.5 animate-cursor" />
                )}
              </p>
            </div>
          )}

          {/* Spinner while waiting for first token */}
          {isStreaming && !hasAnswer && (
            <div className="bg-white border border-gray-200 rounded-2xl rounded-bl-sm px-4 py-3 shadow-sm w-fit">
              <TypingDots />
            </div>
          )}

          {/* Sources shown once done */}
          {!isStreaming && hasSources && (
            <SourcesPanel
              sources={message.sources!}
              latencyMs={message.latencyMs}
              open={sourcesOpen}
              onToggle={() => setSourcesOpen((o) => !o)}
            />
          )}

          {/* Error fallback */}
          {message.trace.error && !isStreaming && (
            <div className="text-xs text-red-600 px-1">
              Something went wrong — try again.
            </div>
          )}
        </div>
      </div>
    )
  }

  // ── Regular message (user or non-traced assistant) ─────────────────────
  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div className={`max-w-[80%] ${isUser ? 'items-end' : 'items-start'} flex flex-col gap-1`}>
        <div
          className={`rounded-2xl px-4 py-3 text-sm leading-relaxed ${
            isUser
              ? 'bg-blue-600 text-white rounded-br-sm'
              : message.error
              ? 'bg-red-50 text-red-800 border border-red-200 rounded-bl-sm'
              : 'bg-white text-gray-800 border border-gray-200 rounded-bl-sm shadow-sm'
          }`}
        >
          <p className="whitespace-pre-wrap">{message.content}</p>
        </div>

        {hasSources && (
          <SourcesPanel
            sources={message.sources!}
            latencyMs={message.latencyMs}
            open={sourcesOpen}
            onToggle={() => setSourcesOpen((o) => !o)}
          />
        )}
      </div>
    </div>
  )
}

// ── Shared sub-components ──────────────────────────────────────────────────

function SourcesPanel({
  sources,
  latencyMs,
  open,
  onToggle,
}: {
  sources: NonNullable<Message['sources']>
  latencyMs?: number
  open: boolean
  onToggle: () => void
}) {
  return (
    <div className="w-full">
      <button
        onClick={onToggle}
        className="flex items-center gap-1 text-xs text-gray-500 hover:text-gray-700 transition-colors py-0.5"
      >
        <svg
          className={`w-3 h-3 transition-transform ${open ? 'rotate-90' : ''}`}
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
          strokeWidth={2.5}
        >
          <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
        </svg>
        {sources.length} source{sources.length !== 1 ? 's' : ''}
        {latencyMs != null && (
          <span className="text-gray-400 ml-1">· {(latencyMs / 1000).toFixed(1)}s</span>
        )}
      </button>

      {open && (
        <div className="mt-1 space-y-1.5 pl-1">
          {sources.map((src, i) => (
            <div
              key={i}
              className="flex items-center gap-2 text-xs bg-gray-50 border border-gray-100 rounded-lg px-3 py-2"
            >
              <svg
                className="w-3 h-3 text-gray-400 shrink-0"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
                strokeWidth={2}
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M7 21h10a2 2 0 002-2V9.414a1 1 0 00-.293-.707l-5.414-5.414A1 1 0
                     0012.586 3H7a2 2 0 00-2 2v14a2 2 0 002 2z"
                />
              </svg>
              <span className="text-gray-700 font-medium truncate">{src.filename}</span>
              {src.page != null && (
                <span className="text-gray-500 shrink-0">p.{src.page}</span>
              )}
              <span className="text-gray-400 shrink-0 ml-auto">
                {src.score.toFixed(2)} relevance
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

function TypingDots() {
  return (
    <div className="flex items-center gap-1 px-1 py-0.5">
      {[0, 1, 2].map((i) => (
        <div
          key={i}
          className="w-2 h-2 rounded-full bg-gray-400 animate-bounce"
          style={{ animationDelay: `${i * 150}ms` }}
        />
      ))}
    </div>
  )
}
