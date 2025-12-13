// This file is used to augment existing types, in this case, the Request object from Express.

declare namespace Express {
  export interface Request {
    user?: {
      userId: string;
      role: string;
    };
  }
}
