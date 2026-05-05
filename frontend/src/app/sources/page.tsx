"use client";
import { DashboardLayout } from "@/components/Dashboard/DashboardLayout";
import { motion } from "framer-motion";
import { Globe, Rss, RefreshCw, CheckCircle, XCircle, Clock } from "lucide-react";
import { useState } from "react";

interface Source {
  id: number;
  name: string;
  type: "api" | "rss" | "scraper";
  url: string;
  status: "active" | "error" | "paused";
  lastFetch: string;
  articlesTotal: number;
}

const initialSources: Source[] = [
  {
    id: 1,
    name: "NewsAPI - Top Headlines",
    type: "api",
    url: "https://newsapi.org/v2/top-headlines",
    status: "active",
    lastFetch: "2 min ago",
    articlesTotal: 42500,
  },
  {
    id: 2,
    name: "HuffPost Archive",
    type: "rss",
    url: "https://www.huffpost.com/section/front-page",
    status: "active",
    lastFetch: "15 min ago",
    articlesTotal: 210000,
  },
  {
    id: 3,
    name: "Reuters World News",
    type: "api",
    url: "https://api.reuters.com/v2/articles",
    status: "active",
    lastFetch: "5 min ago",
    articlesTotal: 87300,
  },
  {
    id: 4,
    name: "TechCrunch Scraper",
    type: "scraper",
    url: "https://techcrunch.com",
    status: "error",
    lastFetch: "3 hours ago",
    articlesTotal: 15200,
  },
  {
    id: 5,
    name: "Reddit r/worldnews",
    type: "api",
    url: "https://www.reddit.com/r/worldnews.json",
    status: "paused",
    lastFetch: "1 day ago",
    articlesTotal: 6800,
  },
];

const statusConfig = {
  active: { color: "text-green-400", icon: CheckCircle, label: "Active", dot: "bg-green-500" },
  error: { color: "text-red-400", icon: XCircle, label: "Error", dot: "bg-red-500" },
  paused: { color: "text-yellow-400", icon: Clock, label: "Paused", dot: "bg-yellow-500" },
};

const typeIcons = {
  api: Globe,
  rss: Rss,
  scraper: RefreshCw,
};

export default function SourcesPage() {
  const [sources, setSources] = useState<Source[]>(initialSources);
  const [refreshing, setRefreshing] = useState<number | null>(null);

  const handleRefresh = (id: number) => {
    setRefreshing(id);
    setTimeout(() => {
      setSources((prev) =>
        prev.map((s) => (s.id === id ? { ...s, lastFetch: "Just now", status: "active" as const } : s))
      );
      setRefreshing(null);
    }, 1500);
  };

  const togglePause = (id: number) => {
    setSources((prev) =>
      prev.map((s) =>
        s.id === id ? { ...s, status: s.status === "paused" ? ("active" as const) : ("paused" as const) } : s
      )
    );
  };

  const activeCount = sources.filter((s) => s.status === "active").length;
  const totalArticles = sources.reduce((sum, s) => sum + s.articlesTotal, 0);

  return (
    <DashboardLayout>
      <div className="space-y-10">
        <section>
          <motion.h1
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            className="text-4xl font-bold mb-2 bg-gradient-to-r from-white to-gray-500 bg-clip-text text-transparent"
          >
            Data Sources
          </motion.h1>
          <p className="text-gray-500">Manage ingestion pipelines and monitor source health.</p>
        </section>

        {/* Stats */}
        <section className="grid grid-cols-3 gap-6">
          {[
            { label: "Total Sources", value: sources.length.toString(), color: "from-purple-500/20 to-purple-500/5" },
            { label: "Active", value: activeCount.toString(), color: "from-green-500/20 to-green-500/5" },
            { label: "Articles Ingested", value: `${(totalArticles / 1000).toFixed(0)}K`, color: "from-blue-500/20 to-blue-500/5" },
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
            </motion.div>
          ))}
        </section>

        {/* Source Table */}
        <section>
          <h2 className="text-xl font-bold mb-6 tracking-tight">All Sources</h2>
          <div className="rounded-2xl border border-white/10 overflow-hidden">
            {/* Header */}
            <div className="grid grid-cols-12 gap-4 px-6 py-3 bg-white/5 text-xs text-gray-500 uppercase tracking-widest font-bold">
              <div className="col-span-4">Source</div>
              <div className="col-span-2">Type</div>
              <div className="col-span-2">Status</div>
              <div className="col-span-2">Last Fetch</div>
              <div className="col-span-2 text-right">Actions</div>
            </div>

            {/* Rows */}
            {sources.map((source, i) => {
              const sCfg = statusConfig[source.status];
              const StatusIcon = sCfg.icon;
              const TypeIcon = typeIcons[source.type];

              return (
                <motion.div
                  key={source.id}
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: i * 0.06 }}
                  className="grid grid-cols-12 gap-4 px-6 py-4 border-t border-white/5 hover:bg-white/[0.03] transition-all items-center"
                >
                  <div className="col-span-4">
                    <p className="font-medium text-sm text-white">{source.name}</p>
                    <p className="text-xs text-gray-600 truncate">{source.url}</p>
                  </div>
                  <div className="col-span-2 flex items-center gap-2 text-sm text-gray-400">
                    <TypeIcon size={14} />
                    <span className="capitalize">{source.type}</span>
                  </div>
                  <div className="col-span-2 flex items-center gap-2">
                    <span className={`w-2 h-2 rounded-full ${sCfg.dot} ${source.status === "active" ? "animate-pulse" : ""}`} />
                    <span className={`text-sm font-medium ${sCfg.color}`}>{sCfg.label}</span>
                  </div>
                  <div className="col-span-2 text-sm text-gray-500">{source.lastFetch}</div>
                  <div className="col-span-2 flex justify-end gap-2">
                    <button
                      onClick={() => handleRefresh(source.id)}
                      disabled={refreshing === source.id}
                      className="px-3 py-1.5 rounded-lg bg-white/5 hover:bg-white/10 text-xs font-medium text-gray-400 hover:text-white transition-all disabled:opacity-30"
                    >
                      <RefreshCw size={12} className={refreshing === source.id ? "animate-spin" : ""} />
                    </button>
                    <button
                      onClick={() => togglePause(source.id)}
                      className="px-3 py-1.5 rounded-lg bg-white/5 hover:bg-white/10 text-xs font-medium text-gray-400 hover:text-white transition-all"
                    >
                      {source.status === "paused" ? "Resume" : "Pause"}
                    </button>
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
