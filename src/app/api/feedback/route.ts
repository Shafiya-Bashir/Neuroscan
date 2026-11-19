import { NextRequest, NextResponse } from "next/server";
import fs from "fs";
import path from "path";

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const { caseId, patientName, aiDiagnosis, action, correctedDiagnosis, clinicianNotes } = body;

    const baseDir = process.cwd();
    const dataDir = path.join(baseDir, "data");
    const logFile = path.join(dataDir, "feedback_audit_log.json");

    if (!fs.existsSync(dataDir)) {
      fs.mkdirSync(dataDir, { recursive: true });
    }

    let existingLogs: any[] = [];
    if (fs.existsSync(logFile)) {
      try {
        const content = fs.readFileSync(logFile, "utf-8");
        existingLogs = JSON.parse(content);
      } catch {
        existingLogs = [];
      }
    }

    const logEntry = {
      auditId: `HITL-${Date.now().toString(36).toUpperCase()}`,
      timestamp: new Date().toISOString(),
      caseId: caseId || "unknown",
      patientName: patientName || "Anonymous",
      aiDiagnosis: aiDiagnosis || "Unspecified",
      clinicianAction: action || "accepted", // 'accepted' | 'corrected' | 'rejected'
      correctedDiagnosis: correctedDiagnosis || null,
      clinicianNotes: clinicianNotes || "",
      activeLearningStatus: action === "corrected" || action === "rejected" ? "QUEUED_FOR_RETRAINING" : "VERIFIED_GOLD_STANDARD"
    };

    existingLogs.unshift(logEntry);
    fs.writeFileSync(logFile, JSON.stringify(existingLogs, null, 2), "utf-8");

    return NextResponse.json({
      success: true,
      message: "Clinician audit record saved to Active Learning Retraining Queue.",
      auditEntry: logEntry
    });
  } catch (error: any) {
    return NextResponse.json(
      { success: false, error: error.message || "Failed to log feedback" },
      { status: 500 }
    );
  }
}

export async function GET() {
  try {
    const logFile = path.join(process.cwd(), "data", "feedback_audit_log.json");
    if (fs.existsSync(logFile)) {
      const content = fs.readFileSync(logFile, "utf-8");
      return NextResponse.json({ success: true, logs: JSON.parse(content) });
    }
    return NextResponse.json({ success: true, logs: [] });
  } catch (error: any) {
    return NextResponse.json({ success: false, logs: [], error: error.message }, { status: 500 });
  }
}
