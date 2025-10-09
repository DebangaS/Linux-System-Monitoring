/**
 * Batch API Utility for Dashboard
 * Author: Member 4
 */
export async function fetchBatchDashboardData(requests) {
  const response = await fetch("/api/v2/dashboard/batch", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(requests),
  });

  return response.json();
}
