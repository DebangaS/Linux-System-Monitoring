/**
 * Advanced Dashboard Management System
 * Author: Member 4
 */

class AdvancedDashboard {
  constructor() {
    this.gridStack = null;
    this.widgets = new Map();
    this.currentTheme = "light";
    this.currentTimeRange = "1h";
    this.autoRefreshEnabled = true;
    this.autoRefreshInterval = 30000; // 30 seconds
    this.widgetTemplates = this.getWidgetTemplates();

    this.initializeDashboard();
    this.setupEventListeners();
    this.startAutoRefresh();

    console.log("Advanced Dashboard initialized");
  }

  /** =====================
   *  INITIALIZATION
   *  ===================== */
  initializeDashboard() {
    this.gridStack = GridStack.init(
      {
        minRow: 1,
        cellHeight: "100px",
        margin: 10,
        resizable: { handles: "se" },
        draggable: { handle: ".widget-header" },
      },
      "#dashboardGrid"
    );

    this.loadDashboardLayout();
    this.initializeDefaultWidgets();
    this.applyTheme(this.currentTheme);
  }

  setupEventListeners() {
    // Add widget
    document
      .getElementById("addWidgetBtn")
      ?.addEventListener("click", () => this.showAddWidgetModal());

    // Save layout
    document
      .getElementById("saveLayoutBtn")
      ?.addEventListener("click", () => this.saveDashboardLayout());

    // Reset layout
    document
      .getElementById("resetLayoutBtn")
      ?.addEventListener("click", () => this.resetDashboardLayout());

    // Time range selection
    document.querySelectorAll("[data-range]").forEach((item) => {
      item.addEventListener("click", (e) => {
        e.preventDefault();
        this.changeTimeRange(e.target.dataset.range);
      });
    });

    // Theme selection
    document.querySelectorAll("[data-theme]").forEach((item) => {
      item.addEventListener("click", (e) => {
        e.preventDefault();
        this.changeTheme(e.target.dataset.theme);
      });
    });

    // Widget control actions (refresh, settings, remove)
    document.addEventListener("click", (e) => {
      const action = e.target.dataset.action;
      if (action) this.handleWidgetAction(action, e.target);
    });

    // Window resize debounce
    window.addEventListener(
      "resize",
      _.debounce(() => this.handleResize(), 250)
    );

    // Keyboard shortcuts
    document.addEventListener("keydown", (e) => this.handleKeyboardShortcuts(e));
  }

  /** =====================
   *  WIDGET DEFINITIONS
   *  ===================== */
  getWidgetTemplates() {
    return {
      "cpu-usage": {
        title: "CPU Usage Chart",
        icon: "fas fa-microchip",
        defaultSize: { w: 6, h: 3 },
        create: (id) => window.chartComponents?.createCPUChart(id),
      },
      "memory-usage": {
        title: "Memory Usage Gauge",
        icon: "fas fa-memory",
        defaultSize: { w: 6, h: 3 },
        create: (id) => window.chartComponents?.createMemoryGauge(id),
      },
      "network-activity": {
        title: "Network Activity",
        icon: "fas fa-network-wired",
        defaultSize: { w: 6, h: 3 },
        create: (id) => window.chartComponents?.createNetworkChart(id),
      },
      "process-table": {
        title: "Process Table",
        icon: "fas fa-list",
        defaultSize: { w: 6, h: 4 },
        type: "table",
      },
    };
  }

  initializeDefaultWidgets() {
    // Example CPU widget
    const cpuChart = window.chartComponents?.createCPUChart("cpuChart");
    if (cpuChart) {
      this.widgets.set("cpu-chart", {
        type: "cpu-usage",
        chart: cpuChart,
        element: document.querySelector('[data-gs-id="cpu-chart"]'),
      });
    }
  }

  /** =====================
   *  WIDGET MANAGEMENT
   *  ===================== */
  showAddWidgetModal() {
    const modal = new bootstrap.Modal(
      document.getElementById("addWidgetModal")
    );
    modal.show();
    document
      .querySelectorAll(".widget-option")
      .forEach((opt) => opt.classList.remove("selected"));
  }

  selectWidgetOption(element) {
    document
      .querySelectorAll(".widget-option")
      .forEach((opt) => opt.classList.remove("selected"));
    element.classList.add("selected");
  }

  addSelectedWidget() {
    const selected = document.querySelector(".widget-option.selected");
    if (!selected) return alert("Please select a widget type");

    const widgetType = selected.dataset.widget;
    const template = this.widgetTemplates[widgetType];
    if (!template) return console.error("Unknown widget type:", widgetType);

    this.addWidget(widgetType, template);
    bootstrap.Modal.getInstance(
      document.getElementById("addWidgetModal")
    ).hide();
  }

  addWidget(widgetType, template) {
    const widgetId = `widget-${Date.now()}`;
    const containerId = `${widgetType}-${Date.now()}`;

    const widgetHTML = this.createWidgetHTML(widgetId, containerId, template);
    const element = this.gridStack.addWidget(widgetHTML, template.defaultSize);

    setTimeout(() => {
      const chart = template.create?.(containerId);
      this.widgets.set(widgetId, {
        type: widgetType,
        chart,
        element,
      });
    }, 100);
  }

  createWidgetHTML(widgetId, containerId, template) {
    return `
      <div class="grid-stack-item" data-gs-id="${widgetId}">
        <div class="grid-stack-item-content widget-card">
          <div class="widget-header">
            <h6><i class="${template.icon}"></i> ${template.title}</h6>
            <div class="widget-controls">
              <button class="btn btn-sm btn-outline-secondary" data-action="refresh"><i class="fas fa-sync"></i></button>
              <button class="btn btn-sm btn-outline-secondary" data-action="settings"><i class="fas fa-cog"></i></button>
              <button class="btn btn-sm btn-outline-danger" data-action="remove"><i class="fas fa-times"></i></button>
            </div>
          </div>
          <div class="widget-content">
            <canvas id="${containerId}"></canvas>
          </div>
        </div>
      </div>`;
  }

  handleWidgetAction(action, element) {
    const widgetElement = element.closest(".grid-stack-item");
    const widgetId = widgetElement?.dataset.gsId;
    if (!widgetId) return;

    switch (action) {
      case "refresh":
        this.refreshWidget(widgetId);
        break;
      case "settings":
        this.showWidgetSettings(widgetId);
        break;
      case "remove":
        this.removeWidget(widgetId);
        break;
    }
  }

  refreshWidget(widgetId) {
    const widget = this.widgets.get(widgetId);
    if (!widget) return;

    widget.element.classList.add("loading");
    setTimeout(() => {
      widget.element.classList.remove("loading");
      console.log(`Refreshed widget: ${widgetId}`);
    }, 1000);
  }

  removeWidget(widgetId) {
    if (!confirm("Remove this widget?")) return;
    const widget = this.widgets.get(widgetId);
    if (widget?.chart) window.chartComponents?.destroyChart(widget.chart);
    this.gridStack.removeWidget(widget.element);
    this.widgets.delete(widgetId);
  }

  /** =====================
   *  THEMES & SETTINGS
   *  ===================== */
  changeTheme(theme) {
    if (theme === "auto")
      theme = window.matchMedia("(prefers-color-scheme: dark)").matches
        ? "dark"
        : "light";
    this.currentTheme = theme;
    this.applyTheme(theme);
    localStorage.setItem("dashboard-theme", theme);
  }

  applyTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    if (window.chartComponents)
      window.chartComponents.updateTheme(theme);
  }

  /** =====================
   *  LAYOUT MANAGEMENT
   *  ===================== */
  saveDashboardLayout() {
    const layout = this.gridStack.save();
    localStorage.setItem(
      "dashboard-layout",
      JSON.stringify({
        layout,
        theme: this.currentTheme,
        timeRange: this.currentTimeRange,
        timestamp: Date.now(),
      })
    );
    this.showToast("Layout saved successfully", "success");
  }

  loadDashboardLayout() {
    try {
      const saved = JSON.parse(localStorage.getItem("dashboard-layout"));
      if (!saved) return;
      this.changeTheme(saved.theme);
      this.changeTimeRange(saved.timeRange);
      console.log("Dashboard layout loaded");
    } catch (err) {
      console.error("Error loading layout", err);
    }
  }

  resetDashboardLayout() {
    if (confirm("Reset dashboard layout?")) {
      localStorage.removeItem("dashboard-layout");
      location.reload();
    }
  }

  /** =====================
   *  UTILS & HELPERS
   *  ===================== */
  changeTimeRange(range) {
    this.currentTimeRange = range;
    this.refreshAllWidgets();
  }

  refreshAllWidgets() {
    this.widgets.forEach((_, id) => this.refreshWidget(id));
  }

  startAutoRefresh() {
    if (this.autoRefreshEnabled)
      setInterval(() => this.refreshAllWidgets(), this.autoRefreshInterval);
  }

  handleResize() {
    this.gridStack?.cellHeight("100px");
    this.widgets.forEach((w) =>
      w.chart?.resize && typeof w.chart.resize === "function"
        ? w.chart.resize()
        : null
    );
  }

  handleKeyboardShortcuts(e) {
    if ((e.ctrlKey || e.metaKey) && e.key === "s") {
      e.preventDefault();
      this.saveDashboardLayout();
    }
    if ((e.ctrlKey || e.metaKey) && e.key === "r") {
      e.preventDefault();
      this.refreshAllWidgets();
    }
    if ((e.ctrlKey || e.metaKey) && e.key === "a") {
      e.preventDefault();
      this.showAddWidgetModal();
    }
  }

  showToast(message, type = "info") {
    const toast = document.createElement("div");
    toast.className = `alert alert-${type} position-fixed top-0 end-0 m-3`;
    toast.innerHTML = `${message}`;
    document.body.appendChild(toast);
    setTimeout(() => toast.remove(), 3000);
  }
}

// Initialize on DOM ready
document.addEventListener("DOMContentLoaded", () => {
  window.advancedDashboard = new AdvancedDashboard();
});
