/**
 * Production Dashboard Main Component
 * Author: Member 4
 */

import React, { useState } from "react";
import { CPUChart, MemoryChart, DiskChart, NetworkChart, Alerts, TrendSummary, PredictionOverlay } from "./charts";
import { DashboardThemeProvider } from "./theme";
import { ExportPanel } from "./export";

export default function DashboardMain() {
  const [theme, setTheme] = useState("light");

  const toggleTheme = () => setTheme(theme === "light" ? "dark" : "light");

  return (
    <DashboardThemeProvider theme={theme}>
      <div className="p-6 space-y-6 bg-background text-foreground min-h-screen transition-colors">
        {/* Header */}
        <div className="flex items-center justify-between">
          <h1 className="text-2xl font-semibold">System Monitor Dashboard</h1>
          <button
            onClick={toggleTheme}
            className="px-4 py-2 rounded-lg border text-sm hover:bg-accent"
          >
            Toggle {theme === "light" ? "Dark" : "Light"} Mode
          </button>
        </div>

        {/* Trends Overview */}
        <TrendSummary />

        {/* Charts Section */}
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
          <CPUChart overlay={<PredictionOverlay metric="cpu" />} />
          <MemoryChart overlay={<PredictionOverlay metric="memory" />} />
          <DiskChart />
          <NetworkChart />
        </div>

        {/* Alerts and Export Section */}
        <Alerts />
        <ExportPanel />
      </div>
    </DashboardThemeProvider>
  );
}
