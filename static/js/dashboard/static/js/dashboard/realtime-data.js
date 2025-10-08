/**
 * Real-time Data Management
 * Author: Member 4
 */

class RealTimeDataManager {
  constructor() {
    this.socket = null;
    this.isConnected = false;
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = 10;
    this.reconnectDelay = 1000;
    this.subscriptions = new Map();
    this.dataBuffer = new Map();
    this.bufferSize = 100;

    // Performance tracking
    this.metrics = {
      messagesReceived: 0,
      lastMessageTime: null,
      averageLatency: 0,
      connectionUptime: 0,
    };

    this.initializeConnection();
    this.setupHeartbeat();
    console.log("Real-time Data Manager initialized");
  }

  initializeConnection() {
    try {
      this.socket = io({
        transports: ["websocket", "polling"],
        timeout: 20000,
        forceNew: true,
      });
      this.setupEventHandlers();
    } catch (error) {
      console.error("Failed to initialize WebSocket connection:", error);
      this.scheduleReconnect();
    }
  }

  setupEventHandlers() {
    this.socket.on("connect", () => {
      console.log("WebSocket connected");
      this.isConnected = true;
      this.reconnectAttempts = 0;
      this.updateConnectionStatus("connected");
      this.metrics.connectionUptime = Date.now();
      this.resubscribeAll();
    });

    this.socket.on("disconnect", (reason) => {
      console.log("WebSocket disconnected:", reason);
      this.isConnected = false;
      this.updateConnectionStatus("disconnected");
      if (reason === "io server disconnect") return;
      this.scheduleReconnect();
    });

    this.socket.on("connect_error", (error) => {
      console.error("WebSocket connection error:", error);
      this.updateConnectionStatus("error");
      this.scheduleReconnect();
    });

    // Stream data handlers
    this.socket.on("stream_message", (data) => this.handleStreamMessage(data));
    this.socket.on("system_metrics", (data) => this.handleSystemMetrics(data));
    this.socket.on("process_data", (data) => this.handleProcessData(data));
    this.socket.on("gpu_data", (data) => this.handleGPUData(data));
    this.socket.on("network_data", (data) => this.handleNetworkData(data));
    this.socket.on("alerts", (data) => this.handleAlerts(data));
    this.socket.on("health_update", (data) => this.handleHealthUpdate(data));
  }

  setupHeartbeat() {
    setInterval(() => {
      if (this.isConnected) {
        const heartbeatTime = Date.now();
        this.socket.emit("heartbeat", { timestamp: heartbeatTime });
      }
    }, 30000);

    this.socket.on("heartbeat_response", (data) => {
      const latency = Date.now() - data.timestamp;
      this.updateLatencyMetrics(latency);
    });
  }

  scheduleReconnect() {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.error("Max reconnection attempts reached");
      this.updateConnectionStatus("failed");
      return;
    }
    const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts);
    this.reconnectAttempts++;
    console.log(
      `Scheduling reconnection attempt ${this.reconnectAttempts} in ${delay}ms`
    );
    this.updateConnectionStatus("reconnecting");
    setTimeout(() => {
      if (!this.isConnected) this.initializeConnection();
    }, delay);
  }

  updateConnectionStatus(status) {
    const statusElement = document.getElementById("connectionStatus");
    if (!statusElement) return;

    const statusConfig = {
      connected: { class: "bg-success", text: "Connected", icon: "fas fa-circle" },
      disconnected: { class: "bg-warning", text: "Disconnected", icon: "fas fa-exclamation" },
      reconnecting: { class: "bg-info", text: "Reconnecting...", icon: "fas fa-sync" },
      error: { class: "bg-danger", text: "Connection Error", icon: "fas fa-times-circle" },
      failed: { class: "bg-danger", text: "Connection Failed", icon: "fas fa-times" },
    };

    const config = statusConfig[status] || statusConfig.disconnected;
    statusElement.className = `badge ${config.class}`;
    statusElement.innerHTML = `<i class="${config.icon}"></i> ${config.text}`;
  }

  updateLatencyMetrics(latency) {
    this.metrics.averageLatency =
      this.metrics.averageLatency === 0
        ? latency
        : this.metrics.averageLatency * 0.9 + latency * 0.1;
  }

  // Subscription management
  subscribe(streamName, callback) {
    if (!this.subscriptions.has(streamName)) {
      this.subscriptions.set(streamName, new Set());
    }
    this.subscriptions.get(streamName).add(callback);
    if (this.isConnected) {
      this.socket.emit("subscribe_stream", { stream_name: streamName });
      console.log(`Subscribed to stream: ${streamName}`);
    }
  }

  unsubscribe(streamName, callback) {
    if (this.subscriptions.has(streamName)) {
      this.subscriptions.get(streamName).delete(callback);
      if (this.subscriptions.get(streamName).size === 0) {
        this.subscriptions.delete(streamName);
        if (this.isConnected) {
          this.socket.emit("unsubscribe_stream", { stream_name: streamName });
        }
      }
    }
    console.log(`Unsubscribed from stream: ${streamName}`);
  }

  resubscribeAll() {
    for (const streamName of this.subscriptions.keys()) {
      this.socket.emit("subscribe_stream", { stream_name: streamName });
    }
  }

  // Data handlers
  handleStreamMessage(data) {
    this.metrics.messagesReceived++;
    this.metrics.lastMessageTime = Date.now();

    const streamName = data.stream_name;
    const message = data.message;

    this.addToBuffer(streamName, message);

    if (this.subscriptions.has(streamName)) {
      this.subscriptions.get(streamName).forEach((callback) => {
        try {
          callback(message);
        } catch (error) {
          console.error("Error in subscription callback:", error);
        }
      });
    }
  }

  handleSystemMetrics(data) {
    if (data.type === "delta") data = this.applyDeltaCompression(data);

    if (data.cpu && window.chartComponents) {
      const cpuData = [{ timestamp: data.timestamp, value: data.cpu.usage_percent || 0 }];
      window.chartComponents.updateCPUChart("cpuChart", cpuData);
    }

    if (data.memory && window.chartComponents) {
      const memoryPercent = data.memory.percent || 0;
      window.chartComponents.updateMemoryGauge("memoryGauge", memoryPercent);
    }

    this.updateQuickStats(data);
  }

  handleProcessData(data) {
    this.updateProcessTable(data);
  }

  handleGPUData(data) {
    this.updateGPUWidgets(data);
  }

  handleNetworkData(data) {
    if (window.chartComponents) {
      window.chartComponents.updateNetworkChart("networkChart", data);
    }
  }

  handleAlerts(data) {
    this.showAlert(data);
    this.updateRecentAlerts(data);
  }

  handleHealthUpdate(data) {
    this.updateHealthScore(data);
  }

  // UI Update methods
  updateQuickStats(data) {
    if (data.cpu) {
      const cpuElement = document.getElementById("cpuQuickStat");
      const cpuTrendElement = document.getElementById("cpuTrend");
      if (cpuElement) cpuElement.textContent = `${(data.cpu.usage_percent || 0).toFixed(1)}%`;
      if (cpuTrendElement) {
        const trend = this.calculateTrend("cpu", data.cpu.usage_percent);
        this.updateTrendIndicator(cpuTrendElement, trend);
      }
    }

    if (data.memory) {
      const memoryElement = document.getElementById("memoryQuickStat");
      const memoryTrendElement = document.getElementById("memoryTrend");
      if (memoryElement) memoryElement.textContent = `${(data.memory.percent || 0).toFixed(1)}%`;
      if (memoryTrendElement) {
        const trend
