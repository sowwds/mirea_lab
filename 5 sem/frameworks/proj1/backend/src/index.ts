const app = require('./app');

const port = process.env.PORT || 3000;

// Start the server only if this file is run directly
if (require.main === module) {
  app.listen(port, () => {
    console.log(`[server]: Server is running at http://localhost:${port}`);
  });
}
