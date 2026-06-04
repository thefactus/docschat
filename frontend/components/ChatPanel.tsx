'use client'

import { useState, useRef, useEffect, useCallback } from 'react'
import { ChatMessage } from '@/components/ChatMessage'
import { sendQuery } from '@/lib/api'
import { activeDriver } from '@/lib/trace-stream'
import { initialTraceState, reduceTraceEvent } from '@/lib/sse-types'
import type { Document, Message } from '@/lib/types'

interface Props {
  documents: Document[]
  selectedDocIds: string[]
}

export function ChatPanel({ documents, selectedDocIds }: Props) {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [pipelineEnabled, setPipelineEnabled] = useState(true)
  const bottomRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const abortRef = useRef<(() => void) | null>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, isLoading])

  // ── Streaming submit (pipeline view ON) ─────────────────────────────
  const submitStream = useCallback(
    (question: string, scope: string[] | null) => {
      const msgId = crypto.randomUUID()

      setMessages((m) => [
        ...m,
        {
          id: msgId,
          role: 'assistant',
          content: '',
          streaming: true,
          trace: initialTraceState(),
        },
      ])

      const { abort } = activeDriver(
        question,
        scope,
        (event) => {
          setMessages((m) =>
            m.map((msg) => {
              if (msg.id !== msgId) return msg
              const newTrace = reduceTraceEvent(msg.trace!, event)

              if (event.event === 'done') {
                return {
                  ...msg,
                  streaming: false,
                  content: newTrace.answer,
                  sources: event.data.sources,
                  tokensUsed: event.data.tokens_used,
                  latencyMs: event.data.latency_ms,
                  trace: newTrace,
                }
              }

              if (event.event === 'error') {
                return {
                  ...msg,
                  streaming: false,
                  error: true,
                  content: event.data.message,
                  trace: newTrace,
                }
              }

              return { ...msg, trace: newTrace }
            }),
          )

          if (event.event === 'done' || event.event === 'error') {
            setIsLoading(false)
            abortRef.current = null
          }
        },
        (errorMsg) => {
          setMessages((m) =>
            m.map((msg) =>
              msg.id !== msgId
                ? msg
                : {
                    ...msg,
                    streaming: false,
                    error: true,
                    content: errorMsg,
                    trace: msg.trace
                      ? { ...msg.trace, error: errorMsg }
                      : undefined,
                  },
            ),
          )
          setIsLoading(false)
          abortRef.current = null
        },
      )

      abortRef.current = abort
    },
    [],
  )

  // ── Plain submit (pipeline view OFF) ────────────────────────────────
  const submitPlain = useCallback(
    async (question: string, scope: string[] | null) => {
      try {
        const res = await sendQuery(question, scope)
        setMessages((m) => [
          ...m,
          {
            id: crypto.randomUUID(),
            role: 'assistant',
            content: res.answer,
            sources: res.sources,
            tokensUsed: res.tokens_used,
            latencyMs: res.latency_ms,
          },
        ])
      } catch (err) {
        setMessages((m) => [
          ...m,
          {
            id: crypto.randomUUID(),
            role: 'assistant',
            content: err instanceof Error ? err.message : 'Something went wrong.',
            error: true,
          },
        ])
      } finally {
        setIsLoading(false)
        textareaRef.current?.focus()
      }
    },
    [],
  )

  const submit = useCallback(async () => {
    const question = input.trim()
    if (!question || isLoading) return

    setInput('')
    setMessages((m) => [
      ...m,
      { id: crypto.randomUUID(), role: 'user', content: question },
    ])
    setIsLoading(true)

    const scope = selectedDocIds.length > 0 ? selectedDocIds : null

    if (pipelineEnabled) {
      submitStream(question, scope)
    } else {
      await submitPlain(question, scope)
      textareaRef.current?.focus()
    }
  }, [input, isLoading, selectedDocIds, pipelineEnabled, submitStream, submitPlain])

  const hasDocuments = documents.length > 0
  const isEmpty = messages.length === 0

  return (
    <main className="flex-1 flex flex-col h-full bg-white overflow-hidden">
      {/* Top bar — scope info + pipeline toggle */}
      <div className="px-6 py-2.5 border-b border-gray-100 bg-gray-50 flex items-center justify-between">
        <div className="flex items-center gap-2">
          {hasDocuments && (
            <>
              <svg
                className="w-3.5 h-3.5 text-gray-400"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
                strokeWidth={2}
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M3 4a1 1 0 011-1h16a1 1 0 011 1v2a1 1 0 01-.293.707L13 13.414V19a1 1
                     0 01-.553.894l-4 2A1 1 0 017 21v-7.586L3.293 6.707A1 1 0 013 6V4z"
                />
              </svg>
              <span className="text-xs text-gray-500">
                {selectedDocIds.length === 0
                  ? `Searching all ${documents.length} document${documents.length !== 1 ? 's' : ''}`
                  : `Searching ${selectedDocIds.length} of ${documents.length} document${
                      documents.length !== 1 ? 's' : ''
                    }`}
              </span>
            </>
          )}
        </div>

        {/* Pipeline view toggle */}
        <button
          onClick={() => setPipelineEnabled((v) => !v)}
          className={`flex items-center gap-1.5 text-xs px-2.5 py-1 rounded-full border transition-colors ${
            pipelineEnabled
              ? 'bg-indigo-50 border-indigo-200 text-indigo-700 hover:bg-indigo-100'
              : 'bg-white border-gray-200 text-gray-500 hover:bg-gray-50'
          }`}
        >
          <span className="font-mono text-[10px]">◈</span>
          Pipeline view
        </button>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-6 py-6 space-y-5">
        {isEmpty && !isLoading ? (
          <div className="flex flex-col items-center justify-center h-full text-center py-16">
            {hasDocuments ? (
              <>
                <div className="w-14 h-14 rounded-2xl bg-blue-50 flex items-center justify-center mb-4">
                  <svg
                    className="w-7 h-7 text-blue-500"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                    strokeWidth={1.5}
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0
                         012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-3 3-3-3z"
                    />
                  </svg>
                </div>
                <h2 className="text-gray-800 font-semibold text-lg mb-1">
                  Ask about your documents
                </h2>
                <p className="text-gray-400 text-sm max-w-xs">
                  Type a question below. Answers are grounded in your uploaded content
                  with inline citations.
                </p>
              </>
            ) : (
              <>
                <div className="w-14 h-14 rounded-2xl bg-gray-100 flex items-center justify-center mb-4">
                  <svg
                    className="w-7 h-7 text-gray-400"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                    strokeWidth={1.5}
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      d="M9 13h6m-3-3v6m5 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0
                         01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                    />
                  </svg>
                </div>
                <h2 className="text-gray-700 font-semibold text-lg mb-1">No documents yet</h2>
                <p className="text-gray-400 text-sm max-w-xs">
                  Upload a PDF, TXT, or MD file using the panel on the left to get started.
                </p>
              </>
            )}
          </div>
        ) : (
          <>
            {messages.map((msg) => (
              <ChatMessage key={msg.id} message={msg} />
            ))}
            {/* Plain-mode loading indicator (pipeline mode shows its own state) */}
            {isLoading && !pipelineEnabled && (
              <div className="flex justify-start">
                <div className="bg-white border border-gray-200 rounded-2xl rounded-bl-sm px-4 py-3 shadow-sm">
                  <TypingDots />
                </div>
              </div>
            )}
          </>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="px-6 py-4 border-t border-gray-100 bg-white">
        <form
          onSubmit={(e) => {
            e.preventDefault()
            submit()
          }}
          className="flex items-end gap-3"
        >
          <textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault()
                submit()
              }
            }}
            placeholder={
              hasDocuments
                ? 'Ask a question… (Enter to send, Shift+Enter for newline)'
                : 'Upload a document to start chatting'
            }
            disabled={!hasDocuments || isLoading}
            rows={1}
            className="flex-1 resize-none rounded-xl border border-gray-300 px-4 py-3 text-sm
                       text-gray-800 placeholder-gray-400 focus:outline-none focus:ring-2
                       focus:ring-blue-500 focus:border-transparent disabled:bg-gray-50
                       disabled:text-gray-400 disabled:cursor-not-allowed leading-relaxed"
            onInput={(e) => {
              const el = e.currentTarget
              el.style.height = 'auto'
              el.style.height = Math.min(el.scrollHeight, 160) + 'px'
            }}
          />
          <button
            type="submit"
            disabled={!input.trim() || !hasDocuments || isLoading}
            className="shrink-0 w-10 h-10 rounded-xl bg-blue-600 hover:bg-blue-700
                       disabled:bg-gray-200 disabled:cursor-not-allowed transition-colors
                       flex items-center justify-center"
          >
            {isLoading && !pipelineEnabled ? (
              <svg className="w-4 h-4 text-white animate-spin" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
              </svg>
            ) : (
              <svg
                className="w-4 h-4 text-white"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
                strokeWidth={2.5}
              >
                <path strokeLinecap="round" strokeLinejoin="round" d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
              </svg>
            )}
          </button>
        </form>
      </div>
    </main>
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
