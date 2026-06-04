'use client'

import { useRef, useState, useCallback } from 'react'
import { uploadDocument } from '@/lib/api'
import type { Document, UploadStatus } from '@/lib/types'

interface Props {
  documents: Document[]
  selectedDocIds: string[]
  onSelectionChange: (ids: string[]) => void
  onDocumentAdded: () => Promise<void>
}

export function DocumentSidebar({
  documents,
  selectedDocIds,
  onSelectionChange,
  onDocumentAdded,
}: Props) {
  const [uploadStatus, setUploadStatus] = useState<UploadStatus>({ kind: 'idle' })
  const [isDragOver, setIsDragOver] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  const processFile = useCallback(
    async (file: File) => {
      const ext = '.' + (file.name.split('.').pop() ?? '').toLowerCase()
      if (!['.pdf', '.txt', '.md'].includes(ext)) {
        setUploadStatus({
          kind: 'error',
          message: `Unsupported type: ${ext || 'unknown'}. Use PDF, TXT, or MD.`,
        })
        return
      }

      setUploadStatus({ kind: 'uploading', filename: file.name })

      // After 600ms assume upload bytes are sent; switch to indexing label.
      const timer = setTimeout(() => {
        setUploadStatus((s) =>
          s.kind === 'uploading' ? { kind: 'indexing', filename: s.filename } : s,
        )
      }, 600)

      try {
        const result = await uploadDocument(file)
        clearTimeout(timer)
        setUploadStatus({ kind: 'done', filename: result.filename, chunks: result.chunks })
        await onDocumentAdded()
        setTimeout(() => setUploadStatus({ kind: 'idle' }), 2500)
      } catch (err) {
        clearTimeout(timer)
        setUploadStatus({
          kind: 'error',
          message: err instanceof Error ? err.message : 'Upload failed',
        })
      }
    },
    [onDocumentAdded],
  )

  function toggleDoc(id: string) {
    onSelectionChange(
      selectedDocIds.includes(id)
        ? selectedDocIds.filter((d) => d !== id)
        : [...selectedDocIds, id],
    )
  }

  const isProcessing =
    uploadStatus.kind === 'uploading' || uploadStatus.kind === 'indexing'

  return (
    <aside className="w-72 shrink-0 bg-slate-900 flex flex-col h-full">
      {/* Header */}
      <div className="px-5 py-4 border-b border-slate-700">
        <div className="flex items-center gap-2">
          <svg
            className="w-5 h-5 text-blue-400"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            strokeWidth={2}
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2
                 0 012 2v8a2 2 0 01-2 2h-5l-3 3-3-3z"
            />
          </svg>
          <span className="text-white font-semibold text-base tracking-tight">DocChat</span>
        </div>
        <p className="text-slate-400 text-xs mt-1">AI-powered document Q&amp;A</p>
      </div>

      {/* Upload area */}
      <div className="px-4 py-4 border-b border-slate-700">
        <p className="text-slate-400 text-xs font-medium uppercase tracking-wider mb-3">
          Documents
        </p>

        <button
          onClick={() => !isProcessing && inputRef.current?.click()}
          onDragOver={(e) => {
            e.preventDefault()
            setIsDragOver(true)
          }}
          onDragLeave={() => setIsDragOver(false)}
          onDrop={(e) => {
            e.preventDefault()
            setIsDragOver(false)
            const file = e.dataTransfer.files[0]
            if (file) processFile(file)
          }}
          disabled={isProcessing}
          className={`w-full border-2 border-dashed rounded-xl p-4 text-center transition-all
            ${isDragOver
              ? 'border-blue-400 bg-blue-950'
              : isProcessing
              ? 'border-slate-600 bg-slate-800 cursor-not-allowed'
              : 'border-slate-600 bg-slate-800 hover:border-blue-500 cursor-pointer'
            }`}
        >
          {uploadStatus.kind === 'idle' && (
            <>
              <svg
                className="w-7 h-7 text-slate-500 mx-auto mb-2"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
                strokeWidth={1.5}
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12"
                />
              </svg>
              <p className="text-slate-300 text-xs font-medium">Drop file or click to upload</p>
              <p className="text-slate-500 text-xs mt-0.5">PDF, TXT, MD · max 20 MB</p>
            </>
          )}

          {uploadStatus.kind === 'uploading' && (
            <div className="flex items-center gap-2 justify-center">
              <Spinner className="text-blue-400" />
              <div className="text-left">
                <p className="text-slate-300 text-xs font-medium">Uploading…</p>
                <p className="text-slate-500 text-xs truncate max-w-[160px]">
                  {uploadStatus.filename}
                </p>
              </div>
            </div>
          )}

          {uploadStatus.kind === 'indexing' && (
            <div className="flex items-center gap-2 justify-center">
              <Spinner className="text-blue-400" />
              <div className="text-left">
                <p className="text-slate-300 text-xs font-medium">Indexing…</p>
                <p className="text-slate-500 text-xs truncate max-w-[160px]">
                  {uploadStatus.filename}
                </p>
              </div>
            </div>
          )}

          {uploadStatus.kind === 'done' && (
            <div className="flex items-center gap-2 justify-center">
              <svg
                className="w-5 h-5 text-green-400 shrink-0"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
                strokeWidth={2.5}
              >
                <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
              </svg>
              <div className="text-left">
                <p className="text-green-300 text-xs font-medium">Indexed!</p>
                <p className="text-slate-500 text-xs">
                  {uploadStatus.chunks} chunk{uploadStatus.chunks !== 1 ? 's' : ''}
                </p>
              </div>
            </div>
          )}

          {uploadStatus.kind === 'error' && (
            <div className="text-left">
              <p className="text-red-400 text-xs font-medium">Upload failed</p>
              <p className="text-slate-400 text-xs mt-0.5 break-words">{uploadStatus.message}</p>
              <button
                className="text-blue-400 text-xs mt-1.5 hover:underline"
                onClick={(e) => {
                  e.stopPropagation()
                  setUploadStatus({ kind: 'idle' })
                }}
              >
                Try again
              </button>
            </div>
          )}
        </button>

        <input
          ref={inputRef}
          type="file"
          accept=".pdf,.txt,.md"
          className="hidden"
          onChange={(e) => {
            const file = e.target.files?.[0]
            if (file) processFile(file)
            e.target.value = ''
          }}
        />
      </div>

      {/* Document list */}
      <div className="flex-1 overflow-y-auto px-4 py-3 space-y-1">
        {documents.length === 0 ? (
          <p className="text-slate-500 text-xs text-center py-6">No documents yet</p>
        ) : (
          <>
            <p className="text-slate-500 text-xs mb-2">
              {selectedDocIds.length === 0
                ? 'Searching all documents'
                : `Filtering to ${selectedDocIds.length} selected`}
            </p>
            {documents.map((doc) => {
              const selected = selectedDocIds.includes(doc.document_id)
              return (
                <button
                  key={doc.document_id}
                  onClick={() => toggleDoc(doc.document_id)}
                  className={`w-full flex items-start gap-2.5 px-3 py-2.5 rounded-lg text-left
                    transition-colors ${
                      selected
                        ? 'bg-blue-900/50 hover:bg-blue-900/70'
                        : 'hover:bg-slate-800'
                    }`}
                >
                  <div
                    className={`mt-0.5 w-4 h-4 rounded border shrink-0 flex items-center
                      justify-center transition-colors ${
                        selected ? 'bg-blue-500 border-blue-500' : 'border-slate-500'
                      }`}
                  >
                    {selected && (
                      <svg
                        className="w-2.5 h-2.5 text-white"
                        fill="none"
                        viewBox="0 0 24 24"
                        stroke="currentColor"
                        strokeWidth={3}
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          d="M5 13l4 4L19 7"
                        />
                      </svg>
                    )}
                  </div>
                  <div className="min-w-0">
                    <p className="text-slate-200 text-xs font-medium truncate">
                      {doc.filename}
                    </p>
                    <p className="text-slate-500 text-xs">
                      {doc.chunks} chunk{doc.chunks !== 1 ? 's' : ''}
                    </p>
                  </div>
                </button>
              )
            })}
          </>
        )}
      </div>
    </aside>
  )
}

function Spinner({ className }: { className?: string }) {
  return (
    <svg
      className={`w-4 h-4 animate-spin shrink-0 ${className ?? ''}`}
      fill="none"
      viewBox="0 0 24 24"
    >
      <circle
        className="opacity-25"
        cx="12"
        cy="12"
        r="10"
        stroke="currentColor"
        strokeWidth="4"
      />
      <path
        className="opacity-75"
        fill="currentColor"
        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
      />
    </svg>
  )
}
