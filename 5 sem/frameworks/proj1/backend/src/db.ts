const { Pool } = require('pg');
require('dotenv').config();

const pool = new Pool({
  connectionString: process.env.DATABASE_URL,
});

module.exports = {
  query: (text: string, params: any[]) => pool.query(text, params),
  pool: pool,
};
