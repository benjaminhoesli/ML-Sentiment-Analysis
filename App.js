import React, { useState } from 'react';
import axios from 'axios';
import TextField from '@mui/material/TextField';
import Button from '@mui/material/Button';
import CircularProgress from '@mui/material/CircularProgress';
import Typography from '@mui/material/Typography';

function App() {
  const [userInput, setUserInput] = useState('');
  const [sentimentPrediction, setSentimentPrediction] = useState('');
  const [loading, setLoading] = useState(false);

  const handleInputChange = (event) => {
    setUserInput(event.target.value);
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    setLoading(true);

    try {
      const response = await axios.post(`http://localhost:5000/sentiment`, {
        user_inp: userInput,
      });
      setSentimentPrediction(response.data);
    } catch (error) {
      console.log(error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ textAlign: 'center', marginTop: '40px' }}>
      <Typography variant="h3">Sentiment Analysis</Typography>
      <form onSubmit={handleSubmit} style={{ marginTop: '20px' }}>
        <TextField
          label="Enter Text"
          variant="outlined"
          value={userInput}
          onChange={handleInputChange}
        />
        <Button
          variant="contained"
          color="primary"
          style={{ marginLeft: '10px' }}
          onClick={handleSubmit}
          disabled={loading}
        >
          {loading ? <CircularProgress size={24} /> : 'Predict Sentiment'}
        </Button>
      </form>
      {loading ? (
        <Typography variant="body1" style={{ marginTop: '20px' }}>
          Loading...
        </Typography>
      ) : (
        <Typography variant="body1" style={{ marginTop: '20px' }}>
          Predicted Sentiment: {sentimentPrediction}
        </Typography>
      )}
    </div>
  );
}

export default App;
