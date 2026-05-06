"use client";
import { DashboardLayout } from "@/components/Dashboard/DashboardLayout";
import dashboardData from "@/data/dashboard.json";
import { motion } from "framer-motion";
import { TrendingUp, ArrowUpRight } from "lucide-react";

const events = dashboardData.events;
const spikeCount = events.filter((event) => event.type === "spike").length;
const avgConfidence = events.length
  ? Math.round((events.reduce((sum, event) => sum + event.confidence, 0) / events.length) * 100)
  : 0;

const typeConfig: Record<string, { color: string; bg: string; label: string }> = {
  spike: { color: "text-red-400", bg: "bg-red-500/10 border-red-500/20", label: "Spike" },
  emerging: { color: "text-blue-400", bg: "bg-blue-500/10 border-blue-500/20", label: "Emerging" },
  declining: { color: "text-yellow-400", bg: "bg-yellow-500/10 border-yellow-500/20", label: "Declining" },
};

export default function EventsPage() {
  return (
    <DashboardLayout>
      <div className="space-y-10">
        <section>
          <motion.h1
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            className="text-4xl font-bold mb-2 bg-gradient-to-r from-white to-gray-500 bg-clip-text text-transparent"
          >
            Event Detection
          </motion.h1>
          <p className="text-gray-500">AI-detected spikes and emerging shifts in the news stream.</p>
        </section>

        {/* Stats Row */}
        <section className="grid grid-cols-3 gap-6">
          {[
            { label: "Active Events", value: events.length.toString(), sub: "Latest 14-day window", color: "from-purple-500/20 to-purple-500/5" },
            { label: "Spike Detections", value: spikeCount.toString(), sub: "Model velocity threshold", color: "from-red-500/20 to-red-500/5" },
            { label: "Avg Confidence", value: `${avgConfidence}%`, sub: "Across generated events", color: "from-blue-500/20 to-blue-500/5" },
          ].map((stat, i) => (
            <motion.div
              key={stat.label}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.1 }}
              className={`p-6 rounded-2xl border border-white/10 bg-gradient-to-br ${stat.color}`}
            >
              <p className="text-xs text-gray-500 uppercase tracking-widest font-bold mb-2">{stat.label}</p>
              <p className="text-3xl font-bold">{stat.value}</p>
              <p className="text-sm text-gray-500 mt-1">{stat.sub}</p>
            </motion.div>
          ))}
        </section>

        {/* Event Timeline */}
        <section>
          <h2 className="text-xl font-bold mb-6 tracking-tight">Recent Events</h2>
          <div className="space-y-4">
            {events.map((event, i) => {
              const cfg = typeConfig[event.type] || typeConfig.emerging;
              return (
                <motion.div
                  key={event.id}
                  initial={{ opacity: 0, x: -30 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: i * 0.08 }}
                  className="p-6 rounded-2xl border border-white/10 bg-white/5 backdrop-blur-xl hover:border-white/20 transition-all group cursor-pointer"
                >
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="flex items-center gap-3 mb-2">
                        <span className={`px-3 py-1 rounded-full text-xs font-bold border ${cfg.bg} ${cfg.color}`}>
                          {cfg.label}
                        </span>
                        <span className="text-xs text-gray-600">{event.time}</span>
                      </div>
                      <h3 className="text-lg font-semibold text-white mb-1 group-hover:text-purple-300 transition-colors">
                        {event.title}
                      </h3>
                      <p className="text-sm text-gray-400 leading-relaxed">{event.summary}</p>
                      <div className="flex items-center gap-4 mt-3">
                        <span className="text-xs text-gray-500 flex items-center gap-1">
                          <TrendingUp size={12} /> {event.trend}
                        </span>
                        <span className="text-xs text-gray-500">
                          Confidence: <span className="text-white font-bold">{(event.confidence * 100).toFixed(0)}%</span>
                        </span>
                      </div>
                    </div>
                    <ArrowUpRight size={20} className="text-gray-600 group-hover:text-purple-400 transition-colors mt-1" />
                  </div>
                </motion.div>
              );
            })}
          </div>
        </section>
      </div>
    </DashboardLayout>
  );
}
