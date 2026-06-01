import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Layout from "@/components/Layout";
import Mining from "@/pages/Mining";
import Monitor from "@/pages/Monitor";
import Alphas from "@/pages/Alphas";
import AlphaDetail from "@/pages/AlphaDetail";
import Algorithm from "@/pages/Algorithm";
import Settings from "@/pages/Settings";

export default function App() {
  return (
    <Router>
      <Routes>
        <Route element={<Layout />}>
          <Route path="/" element={<Mining />} />
          <Route path="/monitor" element={<Monitor />} />
          <Route path="/alphas" element={<Alphas />} />
          <Route path="/alphas/:id" element={<AlphaDetail />} />
          <Route path="/algorithm" element={<Algorithm />} />
          <Route path="/settings" element={<Settings />} />
        </Route>
      </Routes>
    </Router>
  );
}
