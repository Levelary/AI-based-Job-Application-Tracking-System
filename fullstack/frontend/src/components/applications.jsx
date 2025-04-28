import React, { useState, useEffect, use } from 'react';
import axios from 'axios';

const Applications = () => {
    const [applications, setApplcations] = useState([]);
    const [loading, setLoading] = useState(true);
    const [message, setMessage] = useState('');

    useEffect(() => {
        axios.get('http://localhost:5000/applications')
        .then((res) => {
            setApplcations(res.data);
            setLoading(false);
        })
        .catch((err) => {
            console.error('Error fetching applications: ', err);
            setLoading(false);
        });
    }, []);

  return (
    <div style={styles.container}>
    <h2>Applications:</h2>
    {message && <p>{message}</p>}
    {loading ? ( <p>Loading applications...</p>) 
        : (
        <ul style={styles.list}>
            {applications.map(app => (
            <li key={app.id} style={styles.listItem}>
                <strong>Job ID:</strong> {app.job_id} | <strong>Status:</strong> {app.status}
            </li>
            ))}
        </ul>
        )}
    </div>
  )
}

export default Applications;