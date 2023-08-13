import { Metadata } from 'next'

export const metadata: Metadata = {
}

interface RootLayoutProps {
  children: React.ReactNode
}

export default function RootLayout({ children }: RootLayoutProps) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head />
      <body>
      </body>
    </html>
  )
}
