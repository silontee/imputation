import React from "react"
import type { Metadata } from 'next'
import { Poppins, IBM_Plex_Mono } from 'next/font/google'
import { Toaster } from "@/components/ui/toaster"

import './globals.css'

const poppins = Poppins({
  subsets: ['latin'],
  weight: ['300', '400', '500', '600', '700'],
  variable: '--font-poppins'
})

const ibmPlexMono = IBM_Plex_Mono({
  subsets: ['latin'],
  weight: ['400', '500'],
  variable: '--font-mono'
})

export const metadata: Metadata = {
  title: 'ImputeX - Missing Value Imputation',
  description: 'Advanced missing value imputation tool with live processing',
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="ko">
      <body className={`${poppins.variable} ${ibmPlexMono.variable} font-sans antialiased bg-background text-foreground min-h-screen`}>
        {children}
        <Toaster />
      </body>
    </html>
  )
}
