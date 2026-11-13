import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "NEUROSCAN-AI | Deep Learning Brain MRI Diagnostic & Grad-CAM PACS",
  description: "Explainable AI (XAI) Radiology Workstation for Brain Tumor Diagnosis with Grad-CAM Neural Attention Maps",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen bg-[#050811] text-slate-100 antialiased selection:bg-cyan-500/30 selection:text-cyan-200">
        {children}
      </body>
    </html>
  );
}
