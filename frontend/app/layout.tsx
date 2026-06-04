import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'DocChat',
  description: 'Ask questions about your documents using AI',
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="h-full">
      <body className="h-full antialiased bg-gray-50">{children}</body>
    </html>
  )
}
