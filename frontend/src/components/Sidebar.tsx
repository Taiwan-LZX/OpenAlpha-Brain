import { NavLink } from "react-router-dom";
import {
  Pickaxe,
  Activity,
  Trophy,
  Brain,
  Settings,
} from "lucide-react";
import { cn } from "@/lib/utils";

interface NavItem {
  to: string;
  icon: React.ElementType;
  label: string;
}

const navItems: NavItem[] = [
  { to: "/", icon: Pickaxe, label: "挖掘" },
  { to: "/monitor", icon: Activity, label: "监控" },
  { to: "/alphas", icon: Trophy, label: "结果" },
  { to: "/algorithm", icon: Brain, label: "算法" },
  { to: "/settings", icon: Settings, label: "设置" },
];

export default function Sidebar() {
  return (
    <aside className="flex flex-col items-center w-[72px] h-full bg-m3-surface-container border-r border-m3-outline-variant py-m3-4 gap-1">
      <div className="mb-4 flex items-center justify-center w-10 h-10 rounded-m3-lg bg-m3-primary/10">
        <Pickaxe className="h-5 w-5 text-m3-primary" />
      </div>

      <nav className="flex flex-col items-center gap-1 flex-1">
        {navItems.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.to === "/"}
            className={({ isActive }) =>
              cn(
                "flex flex-col items-center justify-center w-14 h-14 rounded-m3-lg transition-colors group",
                isActive
                  ? "bg-m3-primary/15 text-m3-primary"
                  : "text-m3-on-surface-variant hover:bg-m3-surface-container-high hover:text-m3-on-surface"
              )
            }
          >
            <item.icon className="h-5 w-5" />
            <span className="text-label-sm mt-0.5 leading-none">{item.label}</span>
          </NavLink>
        ))}
      </nav>
    </aside>
  );
}
