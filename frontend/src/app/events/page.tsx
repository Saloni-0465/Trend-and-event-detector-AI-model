"use client";
import { DashboardLayout } from "@/components/Dashboard/DashboardLayout";
import { motion } from "framer-motion";
import { Zap, TrendingUp, ArrowUpRight } from "lucide-react";

const mockEvents = [
  {
    id: 1,
    title: "LLM Efficiency Breakthrough",
    summary: "New architecture reduces inference cost by 90%, triggering a massive spike in AI research headlines.",
    type: "spike",
    confidence: 0.95,
    trend: "Large Language Models",
    time: "2 hours ago",
  },
  {
    id: 2,
    title: "EU Green Aviation Mandate",
    summary: "European Union announces mandatory sustainable fuel quotas for all commercial flights by 2030.",
    type: "emerging",
    confidence: 0.82,
    trend: "Sustainable Aviation Fuel",
    time: "5 hours ago",
  },
  {
    id: 3,
    title: "Digital Yuan Pilot Expansion",
    summary: "China expands CBDC pilot program to 10 additional provinces, covering 300M users.",
    type: "emerging",
    confidence: 0.78,
    trend: "Central Bank Digital Currencies",
    time: "8 hours ago",
  },
  {
    id: 4,
    title: "Autonomous Taxi Incident in Austin",
    summary: "Self-driving taxi involved in minor collision, reigniting safety regulation debates across US states.",
    type: "spike",
    confidence: 0.91,
    trend: "Autonomous Vehicle Ethics",
    time: "12 hours ago",
  },
];

const typeConfig: Record<string, { color: string; bg: string; label: string }> = {
  spike: { color: "text-red-400", bg: "bg-red-500/10 border-red-500/20", label: "⚡ Spike" },
  emerging: { color: "text-blue-400", bg: "bg-blue-500/10 border-blue-500/20", label: "🚀 Emerging" },
  declining: { color: "text-yellow-400", bg: "bg-yellow-500/10 border-yellow-500/20", label: "📉 Declining" },
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
            { label: "Active Events", value: "4", sub: "Last 24h", color: "from-purple-500/20 to-purple-500/5" },
            { label: "Spike Detections", value: "2", sub: "+100% vs yesterday", color: "from-red-500/20 to-red-500/5" },
            { label: "Avg Confidence", value: "86%", sub: "Across all events", color: "from-blue-500/20 to-blue-500/5" },
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
            {mockEvents.map((event, i) => {
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
