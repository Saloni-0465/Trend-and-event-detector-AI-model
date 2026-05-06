"use client";
import { DashboardLayout } from "@/components/Dashboard/DashboardLayout";
import dashboardData from "@/data/dashboard.json";
import { motion } from "framer-motion";
import { Bell, BellRing, Check, X, TrendingUp, AlertTriangle } from "lucide-react";
import { useState } from "react";

interface Alert {
  id: number;
  title: string;
  message: string;
  severity: "critical" | "warning" | "info";
  trend: string;
  time: string;
  read: boolean;
}

const initialAlerts: Alert[] = dashboardData.alerts.map((alert) => ({
  ...alert,
  severity: alert.severity as Alert["severity"],
}));

const severityConfig = {
  critical: { color: "text-red-400", bg: "bg-red-500/10 border-red-500/20", icon: AlertTriangle, dot: "bg-red-500" },
  warning: { color: "text-yellow-400", bg: "bg-yellow-500/10 border-yellow-500/20", icon: BellRing, dot: "bg-yellow-500" },
  info: { color: "text-blue-400", bg: "bg-blue-500/10 border-blue-500/20", icon: Bell, dot: "bg-blue-500" },
};

export default function AlertsPage() {
  const [alerts, setAlerts] = useState<Alert[]>(initialAlerts);

  const markRead = (id: number) => {
    setAlerts((prev) => prev.map((a) => (a.id === id ? { ...a, read: true } : a)));
  };

  const dismiss = (id: number) => {
    setAlerts((prev) => prev.filter((a) => a.id !== id));
  };

  const unreadCount = alerts.filter((a) => !a.read).length;

  return (
    <DashboardLayout>
      <div className="space-y-10">
        <section>
          <motion.h1
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            className="text-4xl font-bold mb-2 bg-gradient-to-r from-white to-gray-500 bg-clip-text text-transparent"
          >
            Alerts
          </motion.h1>
          <p className="text-gray-500">
            Model-generated notifications from {dashboardData.meta.dataset}, latest article date {dashboardData.meta.latest_article_date}.
          </p>
        </section>

        {/* Unread Banner */}
        {unreadCount > 0 && (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="p-4 rounded-2xl border border-purple-500/20 bg-purple-500/10 flex items-center justify-between"
          >
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-full bg-purple-500/20 flex items-center justify-center">
                <BellRing size={20} className="text-purple-400" />
              </div>
              <p className="text-sm">
                You have <span className="font-bold text-white">{unreadCount} unread</span> alert{unreadCount > 1 ? "s" : ""}
              </p>
            </div>
            <button
              onClick={() => setAlerts((prev) => prev.map((a) => ({ ...a, read: true })))}
              className="text-xs font-bold text-purple-400 hover:text-purple-300 uppercase tracking-widest transition-colors"
            >
              Mark all read
            </button>
          </motion.div>
        )}

        {/* Alert List */}
        <section className="space-y-3">
          {alerts.map((alert, i) => {
            const cfg = severityConfig[alert.severity];
            const Icon = cfg.icon;
            return (
              <motion.div
                key={alert.id}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: i * 0.05 }}
                className={`p-5 rounded-2xl border border-white/10 backdrop-blur-xl transition-all group ${
                  alert.read ? "bg-white/[0.02] opacity-60" : "bg-white/5"
                }`}
              >
                <div className="flex items-start gap-4">
                  <div className={`w-10 h-10 rounded-xl ${cfg.bg} border flex items-center justify-center shrink-0 mt-0.5`}>
                    <Icon size={18} className={cfg.color} />
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1">
                      {!alert.read && <span className={`w-2 h-2 rounded-full ${cfg.dot} animate-pulse`} />}
                      <h3 className="font-semibold text-white text-sm">{alert.title}</h3>
                      <span className="text-[10px] text-gray-600 ml-auto shrink-0">{alert.time}</span>
                    </div>
                    <p className="text-sm text-gray-400 leading-relaxed mb-2">{alert.message}</p>
                    <span className="text-xs text-gray-600 flex items-center gap-1">
                      <TrendingUp size={10} /> {alert.trend}
                    </span>
                  </div>
                  <div className="flex gap-1 shrink-0 opacity-0 group-hover:opacity-100 transition-opacity">
                    {!alert.read && (
                      <button
                        onClick={() => markRead(alert.id)}
                        className="w-8 h-8 rounded-lg bg-white/5 hover:bg-white/10 flex items-center justify-center transition-all"
                        title="Mark as read"
                      >
                        <Check size={14} className="text-green-400" />
                      </button>
                    )}
                    <button
                      onClick={() => dismiss(alert.id)}
                      className="w-8 h-8 rounded-lg bg-white/5 hover:bg-red-500/10 flex items-center justify-center transition-all"
                      title="Dismiss"
                    >
                      <X size={14} className="text-gray-500 hover:text-red-400" />
                    </button>
                  </div>
                </div>
              </motion.div>
            );
          })}

          {alerts.length === 0 && (
            <div className="text-center py-20 text-gray-600">
              <Bell size={40} className="mx-auto mb-4 opacity-30" />
              <p className="text-lg font-medium">No alerts</p>
              <p className="text-sm">You&apos;re all caught up.</p>
            </div>
          )}
        </section>
      </div>
    </DashboardLayout>
  );
}
