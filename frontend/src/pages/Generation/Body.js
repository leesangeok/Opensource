import React, { useState } from 'react';
import { TextField, Button, Typography, CircularProgress, Box } from '@mui/material';
import { useNavigate } from 'react-router-dom';
import './body.css'; // CSS 파일로 스타일링
import hknuImage from '../../hknu.jpg'; // 이미지 경로 임포트

const Body = () => {
  const [inputText, setInputText] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [isGenerated, setIsGenerated] = useState(false);

  const navigate = useNavigate();

  const handleInputChange = (event) => {
    setInputText(event.target.value);
  };

  const startGeneration = () => {
    setIsGenerating(true);
    setIsGenerated(false);
    
    setTimeout(() => {
      setIsGenerating(false);
      setIsGenerated(true);
    }, 3000);
  };

  const resetGeneration = () => {
    setIsGenerating(false);
    setIsGenerated(false);
    setInputText('');
  };

  const saveGeneratedImage = () => {
    alert('이미지가 저장되었습니다!');
  };

  const goToMyPage = () => {
    navigate('/mypage');
  };

  return (
    <Box className="Generation-main-section">
      <Typography variant="h4" className="title-text">
        심볼 로고 생성기
      </Typography>

      {!isGenerating && !isGenerated && (
        <TextField
          value={inputText}
          onChange={handleInputChange}
          label="로고 설명 입력"
          variant="outlined"
          fullWidth
          sx={{ marginBottom: 2 }}
          className="input-text"
        />
      )}

      {!isGenerating && !isGenerated && (
        <Button
          variant="contained"
          color="primary"
          onClick={startGeneration}
          sx={{
            backgroundColor: '#1E88E5',
            color: 'white',
            padding: '10px 20px',
            borderRadius: '8px',
            width: '100%',
            maxWidth: '200px',
            '&:hover': {
              backgroundColor: '#1976D2'
            }
          }} // MUI의 sx 속성을 통해 직접 스타일링
        >
          생성 시작
        </Button>
      )}

      {isGenerating && (
        <Box className="progress-container">
          <CircularProgress color="secondary" />
          <Typography variant="body1" className="generating-text">
            이미지 생성 중...
          </Typography>
        </Box>
      )}

      {isGenerated && !isGenerating && (
        <>
          <Box sx={{ marginTop: 3 }}>
            <img src={hknuImage} alt="Generated" className="generated-image" />
          </Box>

          <Box sx={{ display: 'flex', justifyContent: 'center', gap: 2, marginTop: 2 }}>
            <Button
              variant="contained"
              onClick={saveGeneratedImage}
              sx={{
                backgroundColor: '#4CAF50',
                color: 'white',
                padding: '10px 20px',
                borderRadius: '8px',
                maxWidth: '200px',
                '&:hover': {
                  backgroundColor: '#45a049'
                }
              }} // MUI의 sx 속성을 통해 직접 스타일링
            >
              저장하기
            </Button>
            <Button
              variant="contained"
              onClick={resetGeneration}
              sx={{
                backgroundColor: '#9C27B0',
                color: 'white',
                padding: '10px 20px',
                borderRadius: '8px',
                maxWidth: '200px',
                '&:hover': {
                  backgroundColor: '#7B1FA2'
                }
              }} // MUI의 sx 속성을 통해 직접 스타일링
            >
              다시 생성하기
            </Button>
            <Button
              variant="contained"
              onClick={goToMyPage}
              sx={{
                backgroundColor: '#673AB7',
                color: 'white',
                padding: '10px 20px',
                borderRadius: '8px',
                maxWidth: '200px',
                '&:hover': {
                  backgroundColor: '#5E35B1'
                }
              }} // MUI의 sx 속성을 통해 직접 스타일링
            >
              마이페이지로 이동
            </Button>
          </Box>
        </>
      )}
    </Box>
  );
};

export default Body;
