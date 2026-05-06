"use client";
import React from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Search, Bell, Settings, LayoutDashboard, Zap, Activity } from "lucide-react";

const NAV_ITEMS = [
  { label: "Overview", icon: LayoutDashboard, href: "/" },
  { label: "Events", icon: Activity, href: "/events" },
  { label: "Alerts", icon: Bell, href: "/alerts" },
];

const SYSTEM_ITEMS = [
  { label: "Sources", icon: Settings, href: "/sources" },
];

export const DashboardLayout = ({ children }: { children: React.ReactNode }) => {
  const pathname = usePathname();

  return (
    <div className="min-h-screen bg-[#050505] text-white selection:bg-purple-500/30">
      {/* Sidebar */}
      <aside className="fixed left-0 top-0 h-full w-64 border-r border-white/10 bg-black/20 backdrop-blur-3xl z-50">
        <div className="p-8">
          <Link href="/" className="flex items-center gap-2 mb-12 group">
            <div className="w-8 h-8 bg-gradient-to-br from-purple-500 to-blue-500 rounded-lg flex items-center justify-center group-hover:shadow-lg group-hover:shadow-purple-500/20 transition-all">
              <Zap size={18} className="text-white" fill="white" />
            </div>
            <span className="font-bold text-xl tracking-tight">TrendPulse</span>
          </Link>

          <nav className="space-y-6">
            <div className="space-y-2">
              <p className="text-[10px] uppercase tracking-widest text-gray-500 font-bold mb-4">Analytics</p>
              {NAV_ITEMS.map((item) => (
                <NavItem
                  key={item.href}
                  icon={<item.icon size={20} />}
                  label={item.label}
                  href={item.href}
                  active={pathname === item.href}
                />
              ))}
            </div>

            <div className="space-y-2">
              <p className="text-[10px] uppercase tracking-widest text-gray-500 font-bold mb-4">System</p>
              {SYSTEM_ITEMS.map((item) => (
                <NavItem
                  key={item.href}
                  icon={<item.icon size={20} />}
                  label={item.label}
                  href={item.href}
                  active={pathname === item.href}
                />
              ))}
            </div>
          </nav>
        </div>
      </aside>

      {/* Main Content */}
      <main className="ml-64 p-8">
        <header className="flex justify-between items-center mb-12">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500" size={18} />
            <input
              type="text"
              placeholder="Search trends or keywords..."
              className="bg-white/5 border border-white/10 rounded-xl py-2 pl-10 pr-4 w-96 focus:outline-none focus:border-purple-500/50 transition-all"
            />
          </div>
          <div className="flex items-center gap-4">
            <Link
              href="/alerts"
              className="w-10 h-10 rounded-full bg-white/5 border border-white/10 flex items-center justify-center hover:bg-white/10 transition-all relative"
            >
              <Bell size={20} className="text-gray-400" />
              <span className="absolute -top-1 -right-1 w-4 h-4 bg-red-500 rounded-full text-[9px] flex items-center justify-center font-bold">3</span>
            </Link>
            <div className="w-10 h-10 rounded-full bg-gradient-to-br from-gray-700 to-gray-800 border border-white/10" />
          </div>
        </header>

        {children}
      </main>
    </div>
  );
};

const NavItem = ({
  icon,
  label,
  href,
  active = false,
}: {
  icon: React.ReactNode;
  label: string;
  href: string;
  active?: boolean;
}) => (
  <Link
    href={href}
    className={`flex items-center gap-3 px-4 py-2 rounded-xl transition-all cursor-pointer ${
      active
        ? "bg-white/10 text-white border border-white/10"
        : "text-gray-500 hover:text-gray-300 hover:bg-white/5 border border-transparent"
    }`}
  >
    {icon}
    <span className="font-medium text-sm">{label}</span>
  </Link>
);
