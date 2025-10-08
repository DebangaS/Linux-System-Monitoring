/**
 * Advanced Chart Components
 * Author: Member 4
 */

class ChartComponents {
  constructor() {
    this.charts = new Map();
    this.defaultOptions = this.getDefaultChartOptions();
    this.colorSchemes = this.getColorSchemes();
    console.log('Chart Components initialized');
  }

  getDefaultChartOptions() {
    return {
      responsive: true,
      maintainAspectRatio: false,
      interaction: {
        intersect: false,
        mode: 'index',
      },
      plugins: {
        legend: {
          position: 'bottom',
          labels: {
            usePointStyle: true,
            padding: 20,
          },
        },
        tooltip: {
          backgroundColor: 'rgba(0, 0, 0, 0.8)',
          titleColor: '#ffffff',
          bodyColor: '#ffffff',
          borderColor: '#dee2e6',
          borderWidth: 1,
          cornerRadius: 8,
          padding: 12,
        },
      },
      scales: {
        x: {
          grid: { color: 'rgba(0, 0, 0, 0.05)' },
          ticks: { maxTicksLimit: 10 },
        },
        y: {
          grid: { color: 'rgba(0, 0, 0, 0.05)' },
          beginAtZero: true,
        },
      },
      animation: {
        duration: 1000,
        easing: 'easeInOutQuart',
      },
    };
  }

  getColorSchemes() {
    return {
      primary: [
        '#0d6efd', '#dc3545', '#20c997',
        '#6610f2', '#fd7e14', '#0dcaf0',
        '#6f42c1', '#ffc107', '#d63384', '#198754',
      ],
      gradient: [
        'rgba(13,110,253,0.8)', 'rgba(111,66,193,0.8)',
        'rgba(220,53,69,0.8)', 'rgba(102,16,242,0.8)',
        'rgba(214,51,132,0.8)', 'rgba(253,126,20,0.8)',
      ],
      performance: {
        excellent: '#198754',
        good: '#20c997',
        warning: '#ffc107',
        critical: '#dc3545',
      },
    };
  }

  // ==========================
  // CPU Usage Line Chart
  // ==========================
  createCPUChart(containerId, data = []) {
    const ctx = document.getElementById(containerId);
    if (!ctx) return null;

    const chartConfig = {
      type: 'line',
      data: {
        labels: [],
        datasets: [{
          label: 'CPU Usage %',
          data: [],
          borderColor: this.colorSchemes.primary[0],
          backgroundColor: 'rgba(13,110,253,0.1)',
          borderWidth: 2,
          fill: true,
          tension: 0.4,
          pointBackgroundColor: this.colorSchemes.primary[0],
          pointBorderColor: '#ffffff',
          pointBorderWidth: 2,
          pointRadius: 4,
          pointHoverRadius: 6,
        }],
      },
      options: {
        ...this.defaultOptions,
        scales: {
          ...this.defaultOptions.scales,
          y: {
            ...this.defaultOptions.scales.y,
            max: 100,
            ticks: {
              callback: value => value + '%',
            },
          },
        },
        plugins: {
          ...this.defaultOptions.plugins,
          tooltip: {
            ...this.defaultOptions.plugins.tooltip,
            callbacks: {
              label: context => `CPU Usage: ${context.parsed.y.toFixed(1)}%`,
            },
          },
        },
      },
    };

    const chart = new Chart(ctx, chartConfig);
    this.charts.set(containerId, chart);

    if (data.length > 0) this.updateCPUChart(containerId, data);
    return chart;
  }

  updateCPUChart(containerId, data) {
    const chart = this.charts.get(containerId);
    if (!chart) return;
    const labels = data.map(item => new Date(item.timestamp).toLocaleTimeString());
    const values = data.map(item => item.value);
    chart.data.labels = labels;
    chart.data.datasets[0].data = values;
    chart.update('none');
  }

  // ==========================
  // Memory Usage Gauge
  // ==========================
  createMemoryGauge(containerId, value = 0) {
    const element = document.getElementById(containerId);
    if (!element) return null;

    const data = [{
      type: 'indicator',
      mode: 'gauge+number+delta',
      value: value,
      delta: { reference: 50, increasing: { color: '#dc3545' } },
      gauge: {
        axis: { range: [null, 100], tickwidth: 1, tickcolor: '#dee2e6' },
        bar: { color: this.getPerformanceColor(value) },
        bgcolor: 'rgba(248,249,250,0.8)',
        borderwidth: 2,
        bordercolor: '#dee2e6',
        steps: [
          { range: [0, 50], color: 'rgba(25,135,84,0.2)' },
          { range: [50, 80], color: 'rgba(255,193,7,0.2)' },
          { range: [80, 100], color: 'rgba(220,53,69,0.2)' },
        ],
        threshold: {
          line: { color: '#dc3545', width: 4 },
          thickness: 0.75,
          value: 90,
        },
      },
    }];

    const layout = {
      width: 300,
      height: 250,
      margin: { t: 20, r: 20, l: 20, b: 20 },
      paper_bgcolor: 'rgba(0,0,0,0)',
      font: { color: '#495057', family: 'Segoe UI, sans-serif' },
    };

    Plotly.newPlot(containerId, data, layout, { displayModeBar: false, responsive: true });
    return { containerId, type: 'gauge' };
  }

  updateMemoryGauge(containerId, value) {
    const update = {
      value: [value],
      'gauge.bar.color': [this.getPerformanceColor(value)],
    };
    Plotly.restyle(containerId, update, [0]);
  }

  // ==========================
  // Utility Methods
  // ==========================
  getPerformanceColor(value) {
    if (value >= 90) return this.colorSchemes.performance.critical;
    if (value >= 80) return this.colorSchemes.performance.warning;
    if (value >= 60) return this.colorSchemes.performance.good;
    return this.colorSchemes.performance.excellent;
  }

  formatBytes(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }

  destroyChart(containerId) {
    const chart = this.charts.get(containerId);
    if (chart) {
      if (typeof chart.destroy === 'function') chart.destroy();
      else if (chart.containerId) Plotly.purge(chart.containerId);
      this.charts.delete(containerId);
    }
  }

  updateTheme(theme) {
    const textColor = theme === 'dark' ? '#ffffff' : '#495057';
    const gridColor = theme === 'dark' ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.05)';

    this.charts.forEach(chart => {
      if (chart && typeof chart.update === 'function') {
        if (chart.options.scales) {
          if (chart.options.scales.x) {
            chart.options.scales.x.grid.color = gridColor;
            chart.options.scales.x.ticks.color = textColor;
          }
          if (chart.options.scales.y) {
            chart.options.scales.y.grid.color = gridColor;
            chart.options.scales.y.ticks.color = textColor;
          }
        }
        if (chart.options.plugins.legend)
          chart.options.plugins.legend.labels.color = textColor;
        chart.update();
      }
    });
    console.log(`Updated charts for ${theme} theme`);
  }

  exportChart(containerId, format = 'png') {
    const chart = this.charts.get(containerId);
    if (!chart) return null;

    if (typeof chart.toBase64Image === 'function')
      return chart.toBase64Image(format);

    if (chart.containerId)
      return Plotly.toImage(chart.containerId, { format: format, width: 1200, height: 800 });

    return null;
  }
}

// Global instance
window.chartComponents = new ChartComponents();
