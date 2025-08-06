import logging
import os
import platform
import re

import GPUtil
import psutil
import requests
from grafana_api.grafana_face import GrafanaFace
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
from prometheus_client.exposition import basic_auth_handler
from requests.auth import HTTPBasicAuth

# GRAFANA_URL = '103.160.78.156:3000'
# PROMETHUS_URL = '103.160.78.156:9001'
# GRAFANA_API = '103.160.78.156:3000'

GRAFANA_URL = os.getenv("GRAFANA_URL", "207.246.109.178:3005")
PROMETHUS_URL = os.getenv("GRAFANA_URL", "http://207.246.109.178:9090")
GRAFANA_API = os.getenv("GRAFANA_API", "207.246.109.178:3005")

JOB_NAME = os.getenv("JOB_NAME", "job2")
JOB_INTERVAL = int(os.getenv("JOB_INTERVAL", 30))
PUSH_GATEWAY_URL = os.getenv("PUSH_GATEWAY_URL", "http://207.246.109.178:9091")
LOKI_URL = os.getenv("LOKI_URL", "http://207.246.109.178:3100")


class Promethus_Grafana:
    def __init__(self, host_grafana, url_grafana, url_promethus):
        self.username = "admin"
        self.password = "admin123@"
        self.url_grafana = url_grafana
        self.url_promethus = url_promethus
        self.host_grafana = host_grafana

        self.grafana_api = GrafanaFace(
            auth=(self.username, self.password), host=url_grafana
        )

    def promethus_push_to(self, job_name):
        def my_auth_handler(url, method, timeout, headers, data):
            return basic_auth_handler(
                url, method, timeout, headers, data, self.username, self.password
            )

        registry = CollectorRegistry()
        g = Gauge(
            "job_last_success_unixtime",
            "Last time a batch job successfully finished",
            registry=registry,
        )
        g.set_to_current_time()
        push_to_gateway(
            PUSH_GATEWAY_URL, job=job_name, registry=registry, handler=my_auth_handler
        )

    def get_gpu_info(self):
        """
        Lấy thông tin GPU từ hệ thống:
        - macOS: sử dụng lệnh system_profiler SPDisplaysDataType
        - Các hệ điều hành khác: trả về None (sẽ dùng GPUtil sau)
        """
        os_name = platform.system()
        try:
            if os_name == "Darwin":  # macOS
                import subprocess

                command = ["system_profiler", "SPDisplaysDataType"]
                output = subprocess.check_output(command, universal_newlines=True)
                return output
            else:
                return None
        except Exception as e:
            return f"Error retrieving GPU info: {e}"

    def get_num_gpus(self):
        """
        Trả về số lượng GPU hiện có:
        - Nếu macOS: dùng kết quả của get_gpu_info() để đếm số lần xuất hiện 'Chipset Model:'
        - Các hệ điều hành khác: dùng GPUtil (nếu có)
        """
        os_name = platform.system()
        if os_name == "Darwin":
            info = self.get_gpu_info()
            if info:
                # Giả sử mỗi GPU sẽ có dòng "Chipset Model:" trong output
                matches = re.findall(r"Chipset Model:", info)
                return len(matches)
            return 0
        else:
            try:
                import GPUtil

                gpus = GPUtil.getGPUs()
                return len(gpus)
            except Exception:
                return 0

    def generate_gpu_panels(self, job_query, starting_panel_id=9, starting_grid_y=0):
        """
        Tạo ra một danh sách các panel JSON cho từng GPU.
        Mỗi GPU sẽ có 3 panel: Load, Temperature và Memory.

        :param job_query: Giá trị label 'job' dùng trong PromQL
        :param starting_panel_id: ID bắt đầu của panel, sẽ tăng dần theo từng panel
        :param starting_grid_y: Vị trí y ban đầu của panel đầu tiên
        :return: Danh sách panel (JSON snippet) cho các GPU
        """
        num_gpus = self.get_num_gpus()
        panels = []
        panel_id = starting_panel_id
        grid_y = starting_grid_y

        for i in range(num_gpus):
            # Panel GPU Load
            load_panel = {
                "datasource": {"type": "prometheus"},
                "fieldConfig": {
                    "defaults": {
                        "color": {"mode": "palette-classic"},
                        "custom": {
                            "axisBorderShow": False,
                            "axisCenteredZero": False,
                            "axisColorMode": "text",
                            "axisLabel": "",
                            "axisPlacement": "auto",
                            "barAlignment": 0,
                            "barWidthFactor": 0.6,
                            "drawStyle": "line",
                            "fillOpacity": 0,
                            "gradientMode": "none",
                            "hideFrom": {
                                "legend": False,
                                "tooltip": False,
                                "viz": False,
                            },
                            "insertNulls": False,
                            "lineInterpolation": "linear",
                            "lineWidth": 1,
                            "pointSize": 5,
                            "scaleDistribution": {"type": "linear"},
                            "showPoints": "auto",
                            "spanNulls": False,
                            "stacking": {"group": "A", "mode": "none"},
                            "thresholdsStyle": {"mode": "off"},
                        },
                        "mappings": [],
                        "thresholds": {
                            "mode": "absolute",
                            "steps": [
                                {"color": "green", "value": None},
                                {"color": "red", "value": 80},
                            ],
                        },
                    },
                    "overrides": [],
                },
                "gridPos": {"h": 11, "w": 12, "x": 0, "y": grid_y},
                "id": panel_id,
                "options": {
                    "legend": {
                        "calcs": [],
                        "displayMode": "list",
                        "placement": "bottom",
                        "showLegend": True,
                    },
                    "tooltip": {"hideZeros": False, "mode": "single", "sort": "none"},
                },
                "pluginVersion": "11.5.2",
                "targets": [
                    {
                        "disableTextWrap": False,
                        "editorMode": "builder",
                        "expr": f'gpu_{i}_load_percent{{job="{job_query}"}}',
                        "fullMetaSearch": False,
                        "includeNullMetadata": True,
                        "legendFormat": f"GPU {i} Load",
                        "range": True,
                        "refId": "A",
                        "useBackend": False,
                    }
                ],
                "title": f"GPU {i} Load (%)",
                "type": "timeseries",
            }
            panel_id += 1

            # Panel GPU Temperature
            temp_panel = {
                "datasource": {"type": "prometheus"},
                "fieldConfig": {
                    "defaults": {
                        "color": {"mode": "palette-classic"},
                        "custom": {
                            "axisBorderShow": False,
                            "axisCenteredZero": False,
                            "axisColorMode": "text",
                            "axisLabel": "",
                            "axisPlacement": "auto",
                            "barAlignment": 0,
                            "barWidthFactor": 0.6,
                            "drawStyle": "line",
                            "fillOpacity": 0,
                            "gradientMode": "none",
                            "hideFrom": {
                                "legend": False,
                                "tooltip": False,
                                "viz": False,
                            },
                            "insertNulls": False,
                            "lineInterpolation": "linear",
                            "lineWidth": 1,
                            "pointSize": 5,
                            "scaleDistribution": {"type": "linear"},
                            "showPoints": "auto",
                            "spanNulls": False,
                            "stacking": {"group": "A", "mode": "none"},
                            "thresholdsStyle": {"mode": "off"},
                        },
                        "mappings": [],
                        "thresholds": {
                            "mode": "absolute",
                            "steps": [
                                {"color": "green", "value": None},
                                {"color": "red", "value": 80},
                            ],
                        },
                    },
                    "overrides": [],
                },
                "gridPos": {"h": 11, "w": 12, "x": 12, "y": grid_y},
                "id": panel_id,
                "options": {
                    "legend": {
                        "calcs": [],
                        "displayMode": "list",
                        "placement": "bottom",
                        "showLegend": True,
                    },
                    "tooltip": {"hideZeros": False, "mode": "single", "sort": "none"},
                },
                "pluginVersion": "11.5.2",
                "targets": [
                    {
                        "disableTextWrap": False,
                        "editorMode": "builder",
                        "expr": f'gpu_{i}_temperature_celsius{{job="{job_query}"}}',
                        "fullMetaSearch": False,
                        "includeNullMetadata": True,
                        "legendFormat": f"GPU {i} Temp",
                        "range": True,
                        "refId": "A",
                        "useBackend": False,
                    }
                ],
                "title": f"GPU {i} Temperature (°C)",
                "type": "timeseries",
            }
            panel_id += 1

            # Panel GPU Memory (hiển thị cả Memory Used và Total)
            mem_panel = {
                "datasource": {"type": "prometheus"},
                "fieldConfig": {
                    "defaults": {
                        "color": {"mode": "palette-classic"},
                        "custom": {
                            "axisBorderShow": False,
                            "axisCenteredZero": False,
                            "axisColorMode": "text",
                            "axisLabel": "",
                            "axisPlacement": "auto",
                            "barAlignment": 0,
                            "barWidthFactor": 0.6,
                            "drawStyle": "line",
                            "fillOpacity": 0,
                            "gradientMode": "none",
                            "hideFrom": {
                                "legend": False,
                                "tooltip": False,
                                "viz": False,
                            },
                            "insertNulls": False,
                            "lineInterpolation": "linear",
                            "lineWidth": 1,
                            "pointSize": 5,
                            "scaleDistribution": {"type": "linear"},
                            "showPoints": "auto",
                            "spanNulls": False,
                            "stacking": {"group": "A", "mode": "none"},
                            "thresholdsStyle": {"mode": "off"},
                        },
                        "mappings": [],
                        "thresholds": {
                            "mode": "absolute",
                            "steps": [
                                {"color": "green", "value": None},
                                {"color": "red", "value": 80},
                            ],
                        },
                    },
                    "overrides": [],
                },
                # "gridPos": {"h": 11, "w": 24, "x": 0, "y": grid_y + 11},
                "gridPos": {"h": 11, "w": 12, "x": 12, "y": grid_y},
                "id": panel_id,
                "options": {
                    "legend": {
                        "calcs": [],
                        "displayMode": "list",
                        "placement": "bottom",
                        "showLegend": True,
                    },
                    "tooltip": {"hideZeros": False, "mode": "single", "sort": "none"},
                },
                "pluginVersion": "11.5.2",
                "targets": [
                    {
                        "disableTextWrap": False,
                        "editorMode": "builder",
                        "expr": f'gpu_{i}_memory_used_mb{{job="{job_query}"}}',
                        "fullMetaSearch": False,
                        "includeNullMetadata": True,
                        "legendFormat": f"GPU {i} Memory Used",
                        "range": True,
                        "refId": "A",
                        "useBackend": False,
                    },
                    {
                        "disableTextWrap": False,
                        "editorMode": "builder",
                        "expr": f'gpu_{i}_memory_total_mb{{job="{job_query}"}}',
                        "fullMetaSearch": False,
                        "hide": False,
                        "includeNullMetadata": True,
                        "instant": False,
                        "legendFormat": f"GPU {i} Memory Total",
                        "range": True,
                        "refId": "B",
                        "useBackend": False,
                    },
                ],
                "title": f"GPU {i} Memory (MB)",
                "type": "timeseries",
            }
            panel_id += 1

            panels.extend([load_panel, temp_panel, mem_panel])
            # Cập nhật vị trí y cho GPU tiếp theo (mỗi GPU chiếm 22 đơn vị chiều cao)
            grid_y += 22

        return panels

    def generate_flask_panels(self, job_query):
        panel = [
            {
                "datasource": {
                    "type": "prometheus",
                    # #"uid": "aef04y0eye2v4a"
                },
                "description": "",
                "fieldConfig": {
                    "defaults": {
                        "color": {"mode": "palette-classic"},
                        "custom": {
                            "axisBorderShow": False,
                            "axisCenteredZero": False,
                            "axisColorMode": "text",
                            "axisLabel": "",
                            "axisPlacement": "auto",
                            "barAlignment": 0,
                            "barWidthFactor": 0.6,
                            "drawStyle": "line",
                            "fillOpacity": 10,
                            "gradientMode": "none",
                            "hideFrom": {
                                "legend": False,
                                "tooltip": False,
                                "viz": False,
                            },
                            "insertNulls": False,
                            "lineInterpolation": "linear",
                            "lineWidth": 1,
                            "pointSize": 5,
                            "scaleDistribution": {"type": "linear"},
                            "showPoints": "never",
                            "spanNulls": False,
                            "stacking": {"group": "A", "mode": "none"},
                            "thresholdsStyle": {"mode": "off"},
                        },
                        "mappings": [],
                        "thresholds": {
                            "mode": "absolute",
                            "steps": [
                                {"color": "green", "value": None},
                                {"color": "red", "value": 80},
                            ],
                        },
                        "unit": "none",
                    },
                    "overrides": [],
                },
                "gridPos": {"h": 5, "w": 12, "x": 0, "y": 22},
                "id": 5,
                "options": {
                    "legend": {
                        "calcs": ["mean", "lastNotNull", "max", "min"],
                        "displayMode": "table",
                        "placement": "right",
                        "showLegend": True,
                    },
                    "tooltip": {"hideZeros": False, "mode": "multi", "sort": "none"},
                },
                "pluginVersion": "11.5.2",
                "targets": [
                    {
                        "datasource": {
                            "type": "prometheus",
                            # #"uid": "aef04y0eye2v4a"
                        },
                        "expr": f'flask_http_request_duration_seconds_bucket{{job="{job_query}"}}',
                        "format": "time_series",
                        "interval": "",
                        "intervalFactor": 1,
                        "legendFormat": "{{ path }}",
                        "refId": "A",
                    }
                ],
                "title": "Request duration [s] - p50",
                "type": "timeseries",
            },
            {
                "datasource": {
                    "type": "prometheus",
                    # #"uid": "aef04y0eye2v4a"
                },
                "fieldConfig": {
                    "defaults": {
                        "color": {"mode": "palette-classic"},
                        "custom": {
                            "axisBorderShow": False,
                            "axisCenteredZero": False,
                            "axisColorMode": "text",
                            "axisLabel": "",
                            "axisPlacement": "auto",
                            "barAlignment": 0,
                            "barWidthFactor": 0.6,
                            "drawStyle": "line",
                            "fillOpacity": 10,
                            "gradientMode": "none",
                            "hideFrom": {
                                "legend": False,
                                "tooltip": False,
                                "viz": False,
                            },
                            "insertNulls": False,
                            "lineInterpolation": "linear",
                            "lineWidth": 1,
                            "pointSize": 5,
                            "scaleDistribution": {"type": "linear"},
                            "showPoints": "never",
                            "spanNulls": False,
                            "stacking": {"group": "A", "mode": "none"},
                            "thresholdsStyle": {"mode": "off"},
                        },
                        "mappings": [],
                        "thresholds": {
                            "mode": "absolute",
                            "steps": [
                                {"color": "green", "value": None},
                                {"color": "red", "value": 80},
                            ],
                        },
                        "unit": "short",
                    },
                    "overrides": [
                        {
                            "matcher": {"id": "byName", "options": "errors"},
                            "properties": [
                                {
                                    "id": "color",
                                    "value": {"fixedColor": "#c15c17", "mode": "fixed"},
                                }
                            ],
                        }
                    ],
                },
                "gridPos": {"h": 10, "w": 6, "x": 12, "y": 22},
                "id": 6,
                "options": {
                    "legend": {
                        "calcs": ["mean", "lastNotNull", "max"],
                        "displayMode": "list",
                        "placement": "bottom",
                        "showLegend": True,
                    },
                    "tooltip": {"hideZeros": False, "mode": "multi", "sort": "none"},
                },
                "pluginVersion": "11.5.2",
                "targets": [
                    {
                        "datasource": {
                            "type": "prometheus",
                            # #"uid": "aef04y0eye2v4a"
                        },
                        "expr": f'flask_http_request_duration_seconds_count{{status!="200", job="{job_query}"}}',
                        "format": "time_series",
                        "interval": "",
                        "intervalFactor": 1,
                        "legendFormat": "errors",
                        "refId": "A",
                    }
                ],
                "title": "Errors per second",
                "type": "timeseries",
            },
            {
                "datasource": {
                    "type": "prometheus",
                    # #"uid": "aef04y0eye2v4a"
                },
                "fieldConfig": {
                    "defaults": {
                        "color": {"mode": "palette-classic"},
                        "custom": {
                            "axisBorderShow": False,
                            "axisCenteredZero": False,
                            "axisColorMode": "text",
                            "axisLabel": "",
                            "axisPlacement": "auto",
                            "barAlignment": 0,
                            "barWidthFactor": 0.6,
                            "drawStyle": "bars",
                            "fillOpacity": 100,
                            "gradientMode": "none",
                            "hideFrom": {
                                "legend": False,
                                "tooltip": False,
                                "viz": False,
                            },
                            "insertNulls": False,
                            "lineInterpolation": "linear",
                            "lineWidth": 1,
                            "pointSize": 5,
                            "scaleDistribution": {"type": "linear"},
                            "showPoints": "never",
                            "spanNulls": False,
                            "stacking": {"group": "A", "mode": "normal"},
                            "thresholdsStyle": {"mode": "off"},
                        },
                        "mappings": [],
                        "min": 0,
                        "thresholds": {
                            "mode": "absolute",
                            "steps": [
                                {"color": "green", "value": None},
                                {"color": "red", "value": 80},
                            ],
                        },
                        "unit": "short",
                    },
                    "overrides": [
                        {
                            "matcher": {"id": "byName", "options": "HTTP 500"},
                            "properties": [
                                {
                                    "id": "color",
                                    "value": {"fixedColor": "#bf1b00", "mode": "fixed"},
                                }
                            ],
                        }
                    ],
                },
                "gridPos": {"h": 10, "w": 6, "x": 18, "y": 22},
                "id": 7,
                "options": {
                    "legend": {
                        "calcs": ["mean", "max"],
                        "displayMode": "list",
                        "placement": "bottom",
                        "showLegend": True,
                    },
                    "tooltip": {"hideZeros": False, "mode": "multi", "sort": "none"},
                },
                "pluginVersion": "11.5.2",
                "targets": [
                    {
                        "datasource": {
                            "type": "prometheus",
                            # #"uid": "aef04y0eye2v4a"
                        },
                        "expr": f'flask_http_request_total{{job="{job_query}"}}',
                        "format": "time_series",
                        "interval": "",
                        "intervalFactor": 1,
                        "legendFormat": "HTTP {{ status }}",
                        "refId": "A",
                    }
                ],
                "title": "Total requests per minute",
                "type": "timeseries",
            },
            {
                "datasource": {
                    "type": "prometheus",
                    # #"uid": "aef04y0eye2v4a"
                },
                "fieldConfig": {
                    "defaults": {
                        "color": {"mode": "palette-classic"},
                        "custom": {
                            "axisBorderShow": False,
                            "axisCenteredZero": False,
                            "axisColorMode": "text",
                            "axisLabel": "",
                            "axisPlacement": "auto",
                            "barAlignment": 0,
                            "barWidthFactor": 0.6,
                            "drawStyle": "line",
                            "fillOpacity": 10,
                            "gradientMode": "none",
                            "hideFrom": {
                                "legend": False,
                                "tooltip": False,
                                "viz": False,
                            },
                            "insertNulls": False,
                            "lineInterpolation": "linear",
                            "lineWidth": 1,
                            "pointSize": 5,
                            "scaleDistribution": {"type": "linear"},
                            "showPoints": "never",
                            "spanNulls": False,
                            "stacking": {"group": "A", "mode": "none"},
                            "thresholdsStyle": {"mode": "off"},
                        },
                        "mappings": [],
                        "thresholds": {
                            "mode": "absolute",
                            "steps": [
                                {"color": "green", "value": None},
                                {"color": "red", "value": 80},
                            ],
                        },
                        "unit": "short",
                    },
                    "overrides": [],
                },
                "gridPos": {"h": 5, "w": 12, "x": 0, "y": 27},
                "id": 8,
                "options": {
                    "legend": {
                        "calcs": ["mean", "lastNotNull"],
                        "displayMode": "table",
                        "placement": "right",
                        "showLegend": True,
                    },
                    "tooltip": {"hideZeros": False, "mode": "multi", "sort": "none"},
                },
                "pluginVersion": "11.5.2",
                "targets": [
                    {
                        "datasource": {
                            "type": "prometheus",
                            # #"uid": "aef04y0eye2v4a"
                        },
                        "expr": f'flask_http_request_duration_seconds_count{{job="{job_query}"}}',
                        "format": "time_series",
                        "interval": "",
                        "intervalFactor": 1,
                        "legendFormat": "{{ path }}",
                        "refId": "A",
                    }
                ],
                "title": "Requests per second",
                "type": "timeseries",
            },
        ]
        return panel

    def create_dashboard(self, name_dashboard, title, job_query, tag):
        gpu_panels = self.generate_gpu_panels(job_query)
        flask_panels = self.generate_flask_panels(job_query)
        payload = {
            "dashboard": {
                "annotations": {
                    "list": [
                        {
                            "builtIn": 1,
                            "datasource": {
                                "type": "grafana",
                                # #"uid": "-- Grafana --"
                            },
                            "enable": True,
                            "hide": True,
                            "iconColor": "rgba(0, 211, 255, 1)",
                            "name": "Annotations & Alerts",
                            "type": "dashboard",
                        }
                    ]
                },
                "editable": True,
                "fiscalYearStartMonth": 0,
                "graphTooltip": 0,
                "links": [],
                "panels": [
                    {
                        "datasource": {
                            "type": "prometheus",
                            # #"uid": "aef04y0eye2v4a"
                        },
                        "fieldConfig": {
                            "defaults": {
                                "color": {"mode": "palette-classic"},
                                "custom": {
                                    "axisBorderShow": False,
                                    "axisCenteredZero": False,
                                    "axisColorMode": "text",
                                    "axisLabel": "",
                                    "axisPlacement": "auto",
                                    "barAlignment": 0,
                                    "barWidthFactor": 0.6,
                                    "drawStyle": "line",
                                    "fillOpacity": 0,
                                    "gradientMode": "none",
                                    "hideFrom": {
                                        "legend": False,
                                        "tooltip": False,
                                        "viz": False,
                                    },
                                    "insertNulls": False,
                                    "lineInterpolation": "linear",
                                    "lineWidth": 1,
                                    "pointSize": 5,
                                    "scaleDistribution": {"type": "linear"},
                                    "showPoints": "auto",
                                    "spanNulls": False,
                                    "stacking": {"group": "A", "mode": "none"},
                                    "thresholdsStyle": {"mode": "off"},
                                },
                                "mappings": [],
                                "thresholds": {
                                    "mode": "absolute",
                                    "steps": [
                                        {"color": "green", "value": None},
                                        {"color": "red", "value": 80},
                                    ],
                                },
                            },
                            "overrides": [],
                        },
                        "gridPos": {"h": 11, "w": 12, "x": 0, "y": 0},
                        "id": 1,
                        "options": {
                            "legend": {
                                "calcs": [],
                                "displayMode": "list",
                                "placement": "bottom",
                                "showLegend": True,
                            },
                            "tooltip": {
                                "hideZeros": False,
                                "mode": "single",
                                "sort": "none",
                            },
                        },
                        "pluginVersion": "11.5.2",
                        "targets": [
                            {
                                "datasource": {
                                    "type": "prometheus",
                                    # "uid": "aef04y0eye2v4a"
                                },
                                "disableTextWrap": False,
                                "editorMode": "builder",
                                "expr": f'disk_used_gb{{job="{job_query}"}}',
                                "fullMetaSearch": False,
                                "includeNullMetadata": True,
                                "legendFormat": "__auto",
                                "range": True,
                                "refId": "A",
                                "useBackend": False,
                            },
                            {
                                "datasource": {
                                    "type": "prometheus",
                                    # "uid": "aef04y0eye2v4a"
                                },
                                "disableTextWrap": False,
                                "editorMode": "builder",
                                "expr": f'disk_used_gb{{job="{job_query}"}}',
                                "fullMetaSearch": False,
                                "hide": False,
                                "includeNullMetadata": True,
                                "instant": False,
                                "legendFormat": "__auto",
                                "range": True,
                                "refId": "B",
                                "useBackend": False,
                            },
                        ],
                        "title": "Disk (Gb)",
                        "type": "timeseries",
                    },
                    {
                        "datasource": {
                            "type": "prometheus",
                            # "uid": "aef04y0eye2v4a"
                        },
                        "fieldConfig": {
                            "defaults": {
                                "color": {"mode": "palette-classic"},
                                "custom": {
                                    "axisBorderShow": False,
                                    "axisCenteredZero": False,
                                    "axisColorMode": "text",
                                    "axisLabel": "",
                                    "axisPlacement": "auto",
                                    "barAlignment": 0,
                                    "barWidthFactor": 0.6,
                                    "drawStyle": "line",
                                    "fillOpacity": 0,
                                    "gradientMode": "none",
                                    "hideFrom": {
                                        "legend": False,
                                        "tooltip": False,
                                        "viz": False,
                                    },
                                    "insertNulls": False,
                                    "lineInterpolation": "linear",
                                    "lineWidth": 1,
                                    "pointSize": 5,
                                    "scaleDistribution": {"type": "linear"},
                                    "showPoints": "auto",
                                    "spanNulls": False,
                                    "stacking": {"group": "A", "mode": "none"},
                                    "thresholdsStyle": {"mode": "off"},
                                },
                                "mappings": [],
                                "thresholds": {
                                    "mode": "absolute",
                                    "steps": [
                                        {"color": "green", "value": None},
                                        {"color": "red", "value": 80},
                                    ],
                                },
                            },
                            "overrides": [],
                        },
                        "gridPos": {"h": 11, "w": 12, "x": 12, "y": 0},
                        "id": 2,
                        "options": {
                            "legend": {
                                "calcs": [],
                                "displayMode": "list",
                                "placement": "bottom",
                                "showLegend": True,
                            },
                            "tooltip": {
                                "hideZeros": False,
                                "mode": "single",
                                "sort": "none",
                            },
                        },
                        "pluginVersion": "11.5.2",
                        "targets": [
                            {
                                "datasource": {
                                    "type": "prometheus",
                                    # "uid": "aef04y0eye2v4a"
                                },
                                "disableTextWrap": False,
                                "editorMode": "builder",
                                "expr": f'ram_total_gb{{job="{job_query}"}}',
                                "fullMetaSearch": False,
                                "includeNullMetadata": True,
                                "legendFormat": "__auto",
                                "range": True,
                                "refId": "A",
                                "useBackend": False,
                            },
                            {
                                "datasource": {
                                    "type": "prometheus",
                                    # "uid": "aef04y0eye2v4a"
                                },
                                "disableTextWrap": False,
                                "editorMode": "builder",
                                "expr": f'ram_used_gb{{job="{job_query}"}}',
                                "fullMetaSearch": False,
                                "hide": False,
                                "includeNullMetadata": True,
                                "instant": False,
                                "legendFormat": "__auto",
                                "range": True,
                                "refId": "B",
                                "useBackend": False,
                            },
                        ],
                        "title": "RAM (Gb)",
                        "type": "timeseries",
                    },
                    {
                        "datasource": {
                            "type": "prometheus",
                            # "uid": "aef04y0eye2v4a"
                        },
                        "fieldConfig": {
                            "defaults": {
                                "color": {"mode": "palette-classic"},
                                "custom": {
                                    "axisBorderShow": False,
                                    "axisCenteredZero": False,
                                    "axisColorMode": "text",
                                    "axisLabel": "",
                                    "axisPlacement": "auto",
                                    "barAlignment": 0,
                                    "barWidthFactor": 0.6,
                                    "drawStyle": "line",
                                    "fillOpacity": 0,
                                    "gradientMode": "none",
                                    "hideFrom": {
                                        "legend": False,
                                        "tooltip": False,
                                        "viz": False,
                                    },
                                    "insertNulls": False,
                                    "lineInterpolation": "linear",
                                    "lineWidth": 1,
                                    "pointSize": 5,
                                    "scaleDistribution": {"type": "linear"},
                                    "showPoints": "auto",
                                    "spanNulls": False,
                                    "stacking": {"group": "A", "mode": "none"},
                                    "thresholdsStyle": {"mode": "off"},
                                },
                                "mappings": [],
                                "thresholds": {
                                    "mode": "absolute",
                                    "steps": [
                                        {"color": "green", "value": None},
                                        {"color": "red", "value": 80},
                                    ],
                                },
                            },
                            "overrides": [],
                        },
                        "gridPos": {"h": 11, "w": 12, "x": 0, "y": 11},
                        "id": 3,
                        "options": {
                            "legend": {
                                "calcs": [],
                                "displayMode": "list",
                                "placement": "bottom",
                                "showLegend": True,
                            },
                            "tooltip": {
                                "hideZeros": False,
                                "mode": "single",
                                "sort": "none",
                            },
                        },
                        "pluginVersion": "11.5.2",
                        "targets": [
                            {
                                "datasource": {
                                    "type": "prometheus",
                                    # "uid": "aef04y0eye2v4a"
                                },
                                "disableTextWrap": False,
                                "editorMode": "builder",
                                "expr": f'cpu_threads{{job="{job_query}"}}',
                                "fullMetaSearch": False,
                                "includeNullMetadata": True,
                                "legendFormat": "__auto",
                                "range": True,
                                "refId": "A",
                                "useBackend": False,
                            },
                            {
                                "datasource": {
                                    "type": "prometheus",
                                    # "uid": "aef04y0eye2v4a"
                                },
                                "disableTextWrap": False,
                                "editorMode": "builder",
                                "expr": f'cpu_cores{{job="{job_query}"}}',
                                "fullMetaSearch": False,
                                "hide": False,
                                "includeNullMetadata": True,
                                "instant": False,
                                "legendFormat": "__auto",
                                "range": True,
                                "refId": "B",
                                "useBackend": False,
                            },
                            {
                                "datasource": {
                                    "type": "prometheus",
                                    # "uid": "aef04y0eye2v4a"
                                },
                                "disableTextWrap": False,
                                "editorMode": "builder",
                                "expr": f'cpu_usage_percent{{job="{job_query}"}}',
                                "fullMetaSearch": False,
                                "hide": False,
                                "includeNullMetadata": True,
                                "instant": False,
                                "legendFormat": "__auto",
                                "range": True,
                                "refId": "C",
                                "useBackend": False,
                            },
                        ],
                        "title": "CPU",
                        "type": "timeseries",
                    },
                    {
                        "datasource": {"type": "loki", "uid": "aegwmti85ezggb"},
                        "fieldConfig": {"defaults": {}, "overrides": []},
                        "gridPos": {"h": 11, "w": 12, "x": 12, "y": 11},
                        "id": 4,
                        "options": {
                            "dedupStrategy": "none",
                            "enableInfiniteScrolling": False,
                            "enableLogDetails": True,
                            "prettifyLogMessage": False,
                            "showCommonLabels": False,
                            "showLabels": False,
                            "showTime": False,
                            "sortOrder": "Descending",
                            "wrapLogMessage": False,
                        },
                        "pluginVersion": "11.5.2",
                        "targets": [
                            {
                                "datasource": {"type": "loki", "uid": "aegwmti85ezggb"},
                                "direction": "backward",
                                "editorMode": "code",
                                "expr": f'{{job_name="{job_query}"}}',
                                "queryType": "range",
                                "refId": "A",
                            }
                        ],
                        "title": "Logs",
                        "type": "logs",
                    },
                ]
                + flask_panels
                + gpu_panels,
                "preload": True,
                "refresh": "5s",
                # "schemaVersion": 40,
                "tags": ["ml_llama3"],
                "templating": {"list": []},
                "time": {"from": "now-30m", "to": "now"},
                "timepicker": {},
                "timezone": "browser",
                "title": name_dashboard,
                # "uid": "",
                "version": 15,
                # "weekStart": "",
                "tags": tag,
            },
            "overwrite": True,
        }

        url = f"{self.host_grafana}/api/dashboards/db"

        # Headers
        headers = {
            "Accept-Language": "en-US,en;q=0.9,vi;q=0.8",
            "accept": "application/json, text/plain, */*",
        }

        # Thực hiện yêu cầu GET với Basic Auth
        res = requests.post(
            url, headers=headers, json=payload, auth=HTTPBasicAuth("admin", "admin123@")
        )
        print(res)

    def generate_link_public(self, tag):
        try:
            dashboard_tag = self.grafana_api.search.search_dashboards(tag=tag)
            print(dashboard_tag)
            dashboard_uid = dashboard_tag[0]["uid"]
            print(dashboard_uid)

            # URL API để công khai Dashboard
            url = f"{self.host_grafana}/api/dashboards/uid/{dashboard_uid}/public-dashboards"

            # Headers
            headers = {
                "Accept-Language": "en-US,en;q=0.9,vi;q=0.8",
                "accept": "application/json, text/plain, */*",
            }

            # Thực hiện yêu cầu GET với Basic Auth
            requests.post(
                url,
                headers=headers,
                json={"isEnabled": True},
                auth=HTTPBasicAuth("admin", "admin123@"),
            )
            response = requests.get(
                url, headers=headers, auth=HTTPBasicAuth("admin", "admin123@")
            )

            if response.status_code == 200:
                res = response.json()
                link_url = f'http://{GRAFANA_URL}/public-dashboards/{res["accessToken"]}?orgId=1'
            else:
                return False

            return link_url

        except Exception as e:
            logging.error(f"Lỗi khi thêm job: {e}")
        return False

    def collect_gpu_metrics(self, registry):
        os_name = platform.system()
        if os_name == "Darwin":
            # Sử dụng output từ system_profiler
            gpu_info_text = self.get_gpu_info()
            # Ví dụ: trích xuất VRAM (Dynamic, Max): 1536 MB
            # Lưu ý: Kết quả của system_profiler không chứa thông tin load hay nhiệt độ,
            # do đó ta chỉ lấy được thông tin VRAM (tổng bộ nhớ GPU)
            match = re.search(r"VRAM.*:\s*(\d+)\s*MB", gpu_info_text)
            if match:
                vram_total = float(match.group(1))
            else:
                vram_total = 0.0

            # Thiết lập gauge cho GPU thứ 0
            gpu_memory_total = Gauge(
                "gpu_0_memory_total_mb", "GPU 0 Total Memory in MB", registry=registry
            )
            gpu_memory_total.set(vram_total)
            # Các thông tin khác không có sẵn nên đặt giá trị mặc định (0)
            gpu_load = Gauge(
                "gpu_0_load_percent", "GPU 0 Load Percentage", registry=registry
            )
            gpu_load.set(0)
            gpu_memory_used = Gauge(
                "gpu_0_memory_used_mb", "GPU 0 Memory Used in MB", registry=registry
            )
            gpu_memory_used.set(0)
            gpu_temperature = Gauge(
                "gpu_0_temperature_celsius",
                "GPU 0 Temperature in Celsius",
                registry=registry,
            )
            gpu_temperature.set(0)
        else:
            # Trên Linux/Windows (với GPU Nvidia), sử dụng GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                for i, gpu in enumerate(gpus):
                    gpu_load = Gauge(
                        f"gpu_{i}_load_percent",
                        f"GPU {i} Load Percentage",
                        registry=registry,
                    )
                    gpu_load.set(gpu.load * 100)

                    gpu_memory_used = Gauge(
                        f"gpu_{i}_memory_used_mb",
                        f"GPU {i} Memory Used in MB",
                        registry=registry,
                    )
                    gpu_memory_used.set(gpu.memoryUsed)

                    gpu_memory_total = Gauge(
                        f"gpu_{i}_memory_total_mb",
                        f"GPU {i} Total Memory in MB",
                        registry=registry,
                    )
                    gpu_memory_total.set(gpu.memoryTotal)

                    gpu_temperature = Gauge(
                        f"gpu_{i}_temperature_celsius",
                        f"GPU {i} Temperature in Celsius",
                        registry=registry,
                    )
                    gpu_temperature.set(gpu.temperature)

    def collect_system_metrics(self, job_name):
        registry = CollectorRegistry()

        # CPU Metrics
        cpu_usage = Gauge(
            "cpu_usage_percent", "CPU Usage Percentage", registry=registry
        )
        cpu_usage.set(psutil.cpu_percent(interval=1))

        cpu_cores = Gauge(
            "cpu_cores", "Number of Physical CPU Cores", registry=registry
        )
        cpu_cores.set(psutil.cpu_count(logical=False))

        cpu_threads = Gauge(
            "cpu_threads", "Number of Logical CPU Threads", registry=registry
        )
        cpu_threads.set(psutil.cpu_count(logical=True))

        # RAM Metrics
        ram = psutil.virtual_memory()
        ram_total = Gauge("ram_total_gb", "Total RAM in GB", registry=registry)
        ram_total.set(ram.total / (1024**3))

        ram_used = Gauge("ram_used_gb", "Used RAM in GB", registry=registry)
        ram_used.set(ram.used / (1024**3))

        ram_percent = Gauge(
            "ram_usage_percent", "RAM Usage Percentage", registry=registry
        )
        ram_percent.set(ram.percent)

        # Disk Metrics
        disk = psutil.disk_usage("/")
        disk_total = Gauge("disk_total_gb", "Total Disk Space in GB", registry=registry)
        disk_total.set(disk.total / (1024**3))

        disk_used = Gauge("disk_used_gb", "Used Disk Space in GB", registry=registry)
        disk_used.set(disk.used / (1024**3))

        disk_percent = Gauge(
            "disk_usage_percent", "Disk Usage Percentage", registry=registry
        )
        disk_percent.set(disk.percent)

        # GPU Metrics
        self.collect_gpu_metrics(registry)

        push_to_gateway(PUSH_GATEWAY_URL, job=job_name, registry=registry)
        print(
            "Đã push metrics info compute tới",
            f"{PUSH_GATEWAY_URL}/metrics/job/{job_name}",
        )


def push_info_to_dashboard(job_name):
    promethus_grafana = Promethus_Grafana(
        host_grafana=f"http://{GRAFANA_URL}",
        url_grafana=GRAFANA_API,
        url_promethus=PROMETHUS_URL,
    )

    promethus_grafana.collect_system_metrics(job_name)
    print("Pushed metrics to Gateway")


def genarate_dashboard(job_name):
    promethus_grafana = Promethus_Grafana(
        host_grafana=f"http://{GRAFANA_URL}",
        url_grafana=GRAFANA_API,
        url_promethus=PROMETHUS_URL,
    )

    promethus_grafana.create_dashboard(job_name, "Training Job", job_name, [job_name])

    print(promethus_grafana.generate_link_public(job_name))


def get_public_dashboard(job_name):
    promethus_grafana = Promethus_Grafana(
        host_grafana=f"http://{GRAFANA_URL}",
        url_grafana=GRAFANA_API,
        url_promethus=PROMETHUS_URL,
    )

    url = promethus_grafana.generate_link_public(job_name)

    return url


# promethus_grafana = Promethus_Grafana(
#     host_grafana=f"http://{GRAFANA_URL}", url_grafana=GRAFANA_API, url_promethus=PROMETHUS_URL)

# promethus_grafana.create_dashboard(
#     "1123", "Training Job", "1123", ["ml_llama3"])
# print(promethus_grafana.generate_link_public('ml_llama3'))
