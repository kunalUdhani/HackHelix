import fs from 'fs';

const filePath = 'frontend/src/features/dashboard/Dashboard.jsx';
let content = fs.readFileSync(filePath, 'utf-8');

// Replace recharts imports
content = content.replace(
  /import \{\s*AreaChart.*?Cell\s*\} from 'recharts';/s,
  `import graph1 from '../../assets/graphs/graph1_expected_vs_actual.png';
import graph2 from '../../assets/graphs/graph2_detection_status.png';
import graph3 from '../../assets/graphs/graph3_energy_loss_trend.png';
import graph6 from '../../assets/graphs/graph6_monthly_trend.png';`
);

// Replace AreaChart
content = content.replace(
  /<ResponsiveContainer width="100%" height="100%">\s*<AreaChart.*?<\/AreaChart>\s*<\/ResponsiveContainer>/s,
  `<img src={graph1} alt="Expected vs Actual Consumption" className="w-full h-full object-contain" />`
);

// Replace PieChart
content = content.replace(
  /<ResponsiveContainer width="100%" height="100%">\s*<PieChart.*?<\/PieChart>\s*<\/ResponsiveContainer>/s,
  `<img src={graph2} alt="Detected Threats" className="w-full h-full object-contain" />`
);

// Replace BarChart
content = content.replace(
  /<ResponsiveContainer width="100%" height="100%">\s*<BarChart.*?<\/BarChart>\s*<\/ResponsiveContainer>/s,
  `<img src={graph6} alt="Anomalies Distributed" className="w-full h-full object-contain" />`
);

// Replace LineChart
content = content.replace(
  /<ResponsiveContainer width="100%" height="100%">\s*<LineChart.*?<\/LineChart>\s*<\/ResponsiveContainer>/s,
  `<img src={graph3} alt="Energy Loss Trend" className="w-full h-full object-contain" />`
);

fs.writeFileSync(filePath, content);
console.log("Replaced Dashboard charts");
