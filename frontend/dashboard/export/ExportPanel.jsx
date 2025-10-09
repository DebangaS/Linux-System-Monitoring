
/**
 * Dashboard Export Panel (PDF, CSV, JSON)
 * Author: Member 4
 */
import React, { useState } from "react";
import api from "../../api";

export function ExportPanel() {
  const [type, setType] = useState("pdf");
  const [status, setStatus] = useState("");

  function exportData() {
    setStatus("Generating...");
    api
      .exportDashboard(type)
      .then(() => setStatus("Download ready!"))
      .catch(() => setStatus("Export failed!"));
  }

  return (
    <div>
      <label>Export Report:</label>
      <select value={type} onChange={(e) => setType(e.target.value)}>
        <option value="pdf">PDF</option>
        <option value="csv">CSV</option>
        <option value="json">JSON</option>
      </select>
      <button onClick={exportData}>Export</button>
      {status && <span>{status}</span>}
    </div>
  );
}
