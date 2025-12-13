import type { Request, Response, NextFunction } from 'express';
const jwt = require('jsonwebtoken');

const JWT_SECRET = process.env.JWT_SECRET || 'your-super-secret-key';

// Define a custom interface for requests that have the user property
interface RequestWithUser extends Request {
  user?: {
    userId: string;
    role: string;
  };
}

const authenticateToken = (req: RequestWithUser, res: Response, next: NextFunction) => {
  const authHeader = req.headers['authorization'];
  const token = authHeader && authHeader.split(' ')[1]; // Bearer TOKEN

  if (token == null) {
    return res.sendStatus(401); // if there isn't any token
  }

  jwt.verify(token, JWT_SECRET, (err: any, user: any) => {
    if (err) {
      return res.sendStatus(403); // if the token is no longer valid
    }
    req.user = user;
    next();
  });
};

module.exports = { authenticateToken };
