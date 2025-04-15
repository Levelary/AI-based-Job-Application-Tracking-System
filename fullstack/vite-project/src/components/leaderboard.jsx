import { useEffect, useState } from 'react';
import axios from 'axios';
const Leaderboard = () => {
  const [rankedData, setRankedData] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    axios.get('http://localhost:5000/leaderboard')
    .then((res) => {
      setRankedData(res.data);
      setLoading(false);
    })
    .catch((err) => {
      console.error('Error fetching leaderboard: ', err);
      setLoading(false);
    });
  }, []);
  return (
    <div style={styles.container}>
      <h2 style={styles.title}>Candidate Eligibility Leaderboard</h2>
      { loading ? ( <p> Loading leaderboard</p> ) 
      : (
          <table style={styles.table}>
            <thead>
              <tr>
                <th style={styles.headerCell}>Rank</th>
                <th style={styles.headerCell}>Candidate ID</th>
                <th style={styles.headerCell}>Predicted Score</th>
              </tr>
            </thead>
            <tbody>
              {rankedData.map((candidate, index) => (
                <tr key={candidate.ID}>
                  <td style={styles.cell}>{index + 1}</td>
                  <td style={styles.cell}>{candidate.ID}</td>
                  <td style={styles.cell}>{candidate.Predicted_Rank.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )
      }
    </div>
  );
};

const styles = {
  container: {
    padding: '20px',
    fontFamily: 'Arial, sans-serif',
    textAlign: 'center'
  },
  title: {
    fontSize: '24px',
    marginBottom: '20px'
  },
  table: {
    margin: '0 auto',
    borderCollapse: 'collapse',
    width: '80%'
  },
  headerCell: {
    backgroundColor: '#007bff',
    color: 'white',
    padding: '10px',
    border: '1px solid #ccc'
  },
  cell: {
    padding: '10px',
    border: '1px solid #ccc',
    textAlign: 'center'
  }
};

export default Leaderboard