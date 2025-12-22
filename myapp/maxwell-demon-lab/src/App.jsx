import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { Layout } from './components/layout';
import { 
  HomePage, 
  SimulationPage, 
  PhaseDiagramPage, 
  DemonDesignerPage 
} from './pages';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<HomePage />} />
          <Route path="simulation" element={<SimulationPage />} />
          <Route path="phase-diagram" element={<PhaseDiagramPage />} />
          <Route path="demon-designer" element={<DemonDesignerPage />} />
        </Route>
      </Routes>
    </Router>
  );
}

export default App;
