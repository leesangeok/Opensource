import React from 'react';
import 'bootstrap/dist/css/bootstrap.min.css'; // Bootstrap CSS 파일 import
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom'; // Routes와 Route import
import MainPage from './pages/Home/MainPage';
import Generation from './pages/Generation/Generation'; // Generation 페이지 import
import MyPage from './pages/MyPage/MyPage'; // MyPage를 올바른 경로에서 임포트
function App() {
  return (
    <Router>
      <div className="App">
        <Routes> {/* Switch 대신 Routes 사용 */}
          <Route path="/" element={<MainPage />} />
          <Route path="/generation" element={<Generation />} />
          <Route path="/mypage" element={<MyPage />} /> {/* MyPage 컴포넌트 사용 */}
        </Routes>
      </div>
    </Router>
  );
}

export default App;
