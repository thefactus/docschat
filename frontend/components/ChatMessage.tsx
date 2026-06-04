'use client'

import { useState } from 'react'
import type { Message } from '@/lib/types'

export function ChatMessage({ message }: { message: Message }) {
  const [sourcesOpen, setSourcesOpen] = useState(false)
  const isUser = message.role === 'user'
  const hasSources = message.sources && message.sources.length > 0

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div className={`max-w-[80%] ${isUser ? 'items-end' : 'items-start'} flex flex-col gap-1`}>
        {/* Bubble */}
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

        {/* Sources toggle */}
        {hasSources && (
          <div className="w-full">
            <button
              onClick={() => setSourcesOpen((o) => !o)}
              className="flex items-center gap-1 text-xs text-gray-500 hover:text-gray-700
                         transition-colors py-0.5"
            >
              <svg
                className={`w-3 h-3 transition-transform ${sourcesOpen ? 'rotate-90' : ''}`}
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
                strokeWidth={2.5}
              >
                <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
              </svg>
              {message.sources!.length} source{message.sources!.length !== 1 ? 's' : ''}
              {message.latencyMs != null && (
                <span className="text-gray-400 ml-1">
                  · {(message.latencyMs / 1000).toFixed(1)}s
                </span>
              )}
            </button>

            {sourcesOpen && (
              <div className="mt-1 space-y-1.5 pl-1">
                {message.sources!.map((src, i) => (
                  <div
                    key={i}
                    className="flex items-center gap-2 text-xs bg-gray-50 border border-gray-100
                               rounded-lg px-3 py-2"
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
        )}
      </div>
    </div>
  )
}
