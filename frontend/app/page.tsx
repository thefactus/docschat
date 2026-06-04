'use client'

import { useState, useEffect, useCallback } from 'react'
import { DocumentSidebar } from '@/components/DocumentSidebar'
import { ChatPanel } from '@/components/ChatPanel'
import { listDocuments } from '@/lib/api'
import type { Document } from '@/lib/types'

export default function Page() {
  const [documents, setDocuments] = useState<Document[]>([])
  const [selectedDocIds, setSelectedDocIds] = useState<string[]>([])

  const refreshDocuments = useCallback(async () => {
    try {
      const docs = await listDocuments()
      setDocuments(docs)
    } catch {
      // Non-fatal — sidebar shows empty state, user can still upload
    }
  }, [])

  useEffect(() => {
    refreshDocuments()
  }, [refreshDocuments])

  return (
    <div className="flex h-full overflow-hidden">
      <DocumentSidebar
        documents={documents}
        selectedDocIds={selectedDocIds}
        onSelectionChange={setSelectedDocIds}
        onDocumentAdded={refreshDocuments}
      />
      <ChatPanel documents={documents} selectedDocIds={selectedDocIds} />
    </div>
  )
}
