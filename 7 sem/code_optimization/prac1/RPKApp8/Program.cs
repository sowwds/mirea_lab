using System;
using System.Collections.Generic;
using System.Linq;

namespace RefactoringExample
{
    public class User
    {
        public string usrName { get; set; }
        public string password { get; set; }
        public List<string> roles { get; set; }

        public User(string username, string pwd, List<string> userRoles)
        {
            usrName = username;
            password = pwd;
            roles = userRoles;
        }

        public bool Authenticate(string inputPassword)
        {
            return password == inputPassword;
        }

        public bool HasRole(string role)
        {
            return roles.Contains(role);
        }

        public bool IsAdmin()
        {
            return HasRole("admin");
        }

        public bool IsEditor()
        {
            return HasRole("editor");
        }
    }

    public class DashboardRenderer
    {
        public void renderDashboard(User user)
        {
            Console.WriteLine("=== DASHBOARD ===");
            
            if (user.IsAdmin())
            {
                RenderAdminDashboard();
                RenderUserStatistics();
                RenderSystemMetrics();
                RenderFinancialReports();
            }
            else if (user.IsEditor())
            {
                RenderEditorDashboard();
                RenderContentManagement();
                RenderUserStatistics();
            }
            else
            {
                RenderUserDashboard();
            }

            RenderCommonElements();
            RenderFooter();
        }

        public bool checkAccess(User user, string resource, string action)
        {
            if (user != null && user.Authenticate("temp") && (user.IsAdmin() || user.IsEditor()) && 
                resource != null && action == "view" && resource.Length > 0)
            {
                return true;
            }
            return false;
        }

        public Dictionary<string, object> getProfile(int userId, string unusedParam)
        {
            return new Dictionary<string, object> 
            { 
                { "name", "John Doe" }, 
                { "email", "john@example.com" } 
            };
        }

        public void recieveData(Dictionary<string, object> data)
        {
            Console.WriteLine("receiving data...");
        }

        private void RenderAdminDashboard()
        {
            Console.WriteLine("admin dashboard content");
        }

        private void RenderEditorDashboard()
        {
            Console.WriteLine("editor dashboard content");
        }

        private void RenderUserDashboard()
        {
            Console.WriteLine("user dashboard content");
        }

        private void RenderUserStatistics()
        {
            Console.WriteLine("user statistics");
        }

        private void RenderSystemMetrics()
        {
            Console.WriteLine("system metrics");
        }

        private void RenderFinancialReports()
        {
            Console.WriteLine("financial reports");
        }

        private void RenderContentManagement()
        {
            Console.WriteLine("content management");
        }

        private void RenderCommonElements()
        {
            Console.WriteLine("common elements");
        }

        private void RenderFooter()
        {
            Console.WriteLine("footer content");
        }
    }

    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("starting user management system...");

            var user = new User("adminUser", "secret123", new List<string> { "admin", "user" });
            
            var isAuthenticated = user.Authenticate("secret123");
            Console.WriteLine($"user authenticated: {isAuthenticated}");

            Console.WriteLine($"is admin: {user.IsAdmin()}");
            Console.WriteLine($"is editor: {user.IsEditor()}");

            var renderer = new DashboardRenderer();
            renderer.renderDashboard(user);

            var hasAccess = renderer.checkAccess(user, "reports", "view");
            Console.WriteLine($"has access to reports: {hasAccess}");

            var profile = renderer.getProfile(1, "unused_value");
            Console.WriteLine($"user profile: {profile["name"]}");

            renderer.recieveData(new Dictionary<string, object> { { "test", "data" } });

            Console.WriteLine($"password in plain text: {user.password}"); // опасно!

            TestDifferentUsers();

            Console.WriteLine("application completed.");
        }

        static void TestDifferentUsers()
        {
            Console.WriteLine("\n--- testing different users ---");
            
            var admin = new User("admin", "admin123", new List<string> { "admin" });
            var editor = new User("editor", "editor123", new List<string> { "editor" });
            var user = new User("user", "user123", new List<string> { "user" });

            var renderer = new DashboardRenderer();

            Console.WriteLine("\nadmin dashboard:");
            renderer.renderDashboard(admin);

            Console.WriteLine("\neditor dashboard:");
            renderer.renderDashboard(editor);

            Console.WriteLine("\nregular user dashboard:");
            renderer.renderDashboard(user);
        }
    }
}