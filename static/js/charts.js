/**
 * Chart components for data visualization
 * Author: Member 4
 */

class SystemCharts {
  constructor() {
    this.charts = {};
    this.colors = {
      cpu: '#007bff',
      memory: '#28a745',
      disk: '#17a2b8',
      network: '#ffc107'
    };
  }

  // Create CPU usage chart
  createCPUChart(canvasId, data) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    this.charts[canvasId] = new Chart(ctx, {
      type: 'line',
      data: {
        labels: data.map(d => new Date(d.timestamp).toLocaleTimeString()),
        datasets: [{
          label: 'CPU Usage (%)',
          data: data.map(d => d.usage_percent),
          borderColor: this.colors.cpu,
          backgroundColor: this.colors.cpu + '20',
          borderWidth: 2,
          fill: true,
          tension: 0.4
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          y: {
            beginAtZero: true,
            max: 100,
            title: {
              display: true,
              text: 'CPU Usage (%)'
            }
          },
          x: {
            title: {
              display: true,
              text: 'Time'
            }
          }
        },
        plugins: {
          legend: { display: true, position: 'top' },
          tooltip: { mode: 'index', intersect: false }
        }
      }
    });
  }

  // Create Memory usage chart
  createMemoryChart(canvasId, data) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    this.charts[canvasId] = new Chart(ctx, {
      type: 'line',
      data: {
        labels: data.map(d => new Date(d.timestamp).toLocaleTimeString()),
        datasets: [
          {
            label: 'Memory Usage (%)',
            data: data.map(d => d.usage_percent),
            borderColor: this.colors.memory,
            backgroundColor: this.colors.memory + '20',
            borderWidth: 2,
            fill: true,
            tension: 0.4
          },
          {
            label: 'Swap Usage (%)',
            data: data.map(d => d.swap?.usage_percent || 0),
            borderColor: '#dc3545',
            backgroundColor: '#dc354520',
            borderWidth: 2,
            fill: false,
            tension: 0.4
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          y: {
            beginAtZero: true,
            max: 100,
            title: { display: true, text: 'Memory Usage (%)' }
          }
        },
        plugins: {
          legend: { display: true, position: 'top' }
        }
      }
    });
  }

  // Create Disk usage chart
  createDiskChart(canvasId, data) {
    if (!data || data.length === 0) return;
    const ctx = document.getElementById(canvasId).getContext('2d');
    const latestData = data[data.length - 1];
    this.charts[canvasId] = new Chart(ctx, {
      type: 'doughnut',
      data: {
        labels: latestData.partitions.map(p => p.device),
        datasets: [{
          label: 'Disk Usage (%)',
          data: latestData.partitions.map(p => p.usage_percent),
          backgroundColor: [
            '#007bff', '#28a745', '#17a2b8', '#ffc107',
            '#dc3545', '#6f42c1', '#fd7e14', '#20c997'
          ],
          borderWidth: 2
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: true, position: 'right' },
          tooltip: {
            callbacks: {
              label: function (context) {
                const partition = latestData.partitions[context.dataIndex];
                return `${partition.device}: ${partition.usage_percent.toFixed(1)}%`;
              }
            }
          }
        }
      }
    });
  }

  // Create Network I/O chart
  createNetworkChart(canvasId, data) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    this.charts[canvasId] = new Chart(ctx, {
      type: 'line',
      data: {
        labels: data.map(d => new Date(d.timestamp).toLocaleTimeString()),
        datasets: [
          {
            label: 'Sent (KB/s)',
            data: data.map(d => d.sent_rate_kbps),
            borderColor: '#28a745',
            backgroundColor: '#28a74520',
            borderWidth: 2,
            fill: false,
            tension: 0.4
          },
          {
            label: 'Received (KB/s)',
            data: data.map(d => d.recv_rate_kbps),
            borderColor: '#007bff',
            backgroundColor: '#007bff20',
            borderWidth: 2,
            fill: false,
            tension: 0.4
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          y: {
            beginAtZero: true,
            title: { display: true, text: 'Data Rate (KB/s)' }
          }
        },
        plugins: {
          legend: { display: true, position: 'top' }
        }
      }
    });
  }

  // Update chart
  updateChart(canvasId, newData) {
    if (!this.charts[canvasId]) return;
    const chart = this.charts[canvasId];
    const maxPoints = 50;

    chart.data.labels = newData.map(d => new Date(d.timestamp).toLocaleTimeString());

    if (canvasId.includes('cpu')) {
      chart.data.datasets[0].data = newData.map(d => d.usage_percent).slice(-maxPoints);
    } else if (canvasId.includes('memory')) {
      chart.data.datasets[0].data = newData.map(d => d.usage_percent).slice(-maxPoints);
      chart.data.datasets[1].data = newData.map(d => d.swap?.usage_percent || 0).slice(-maxPoints);
    } else if (canvasId.includes('network')) {
      chart.data.datasets[0].data = newData.map(d => d.sent_rate_kbps).slice(-maxPoints);
      chart.data.datasets[1].data = newData.map(d => d.recv_rate_kbps).slice(-maxPoints);
    }

    chart.update('none');
  }

  // Destroy chart
  destroyChart(canvasId) {
    if (this.charts[canvasId]) {
      this.charts[canvasId].destroy();
      delete this.charts[canvasId];
    }
  }
}

// Global instance
const systemCharts = new SystemCharts();
