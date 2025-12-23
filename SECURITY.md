# Security Policy

## Supported Versions

We actively support the following versions of Diet Pandas with security updates:

| Version | Supported          |
| ------- | ------------------ |
| 0.3.x   | :white_check_mark: |
| 0.2.x   | :white_check_mark: |
| < 0.2   | :x:                |

**Note:** We recommend always using the latest stable version for the best security and performance.

## Reporting a Vulnerability

We take the security of Diet Pandas seriously. If you discover a security vulnerability, please follow these steps:

### 📧 Private Disclosure

**Please DO NOT create a public GitHub issue for security vulnerabilities.**

Instead, report security issues privately by:

1. **Email**: Send a detailed report to **luizfernando1012000@gmail.com**
2. **Subject Line**: Use `[SECURITY] Diet Pandas - [Brief Description]`
3. **Include**:
   - A clear description of the vulnerability
   - Steps to reproduce the issue
   - Potential impact and severity
   - Any suggested fixes (if available)
   - Your contact information for follow-up

### 🕐 Response Timeline

- **Initial Response**: Within 48 hours of receiving your report
- **Status Updates**: Every 3-5 business days until resolved
- **Resolution**: We aim to release a patch within 7-14 days for critical issues

### 🔒 What to Expect

**If the vulnerability is accepted:**
- We'll acknowledge your report and work on a fix
- You'll receive updates on our progress
- We'll coordinate a disclosure timeline with you
- Your contribution will be credited (unless you prefer to remain anonymous)
- A security advisory will be published with the fix

**If the vulnerability is declined:**
- We'll explain why it's not considered a security issue
- We may still address it as a regular bug if applicable

## Security Best Practices

When using Diet Pandas:

1. **Keep Updated**: Always use the latest version
2. **Validate Input**: Sanitize file paths and user inputs before passing to Diet Pandas
3. **File Permissions**: Be cautious when reading files with elevated privileges
4. **Dependencies**: Regularly update dependencies (pandas, numpy, polars, psutil)
5. **Untrusted Data**: Be careful when loading CSV/Parquet files from untrusted sources

## Scope

This security policy applies to the Diet Pandas library itself. Security issues in dependencies (pandas, numpy, polars, psutil) should be reported to their respective maintainers.

## Known Security Considerations

- **File Access**: Diet Pandas reads and writes files. Ensure proper file permissions in your environment.
- **Memory Usage**: Large files can consume significant memory. Use chunked reading for very large datasets.
- **Schema Files**: Schema JSON files are loaded and executed. Only load schema files from trusted sources.

Thank you for helping keep Diet Pandas and our users safe! 🐼🥗🔒
