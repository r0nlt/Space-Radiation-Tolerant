#pragma once

#include <string>
#include <optional>
#include <variant>
#include <type_traits>
#include <functional>
#include <memory>
#include <exception>
#include <cassert>
#include <sstream>
#include <iostream>
#include <vector>
#include <mutex>

namespace rad_ml {
namespace error {

/**
 * @brief Simple source location struct (C++17 compatible replacement for std::source_location)
 */
struct SourceLocation {
    const char* file_name;
    int line;
    const char* function_name;
    
    SourceLocation(const char* file = __FILE__, int line = __LINE__, const char* function = __func__)
        : file_name(file), line(line), function_name(function) {}
        
    std::string toString() const {
        std::stringstream ss;
        ss << file_name << ":" << line << " (" << function_name << ")";
        return ss.str();
    }
};

/**
 * @brief Helper macro to create source location
 */
#define RAD_ML_CURRENT_LOCATION rad_ml::error::SourceLocation(__FILE__, __LINE__, __func__)

/**
 * @brief Error severity levels for the framework
 */
enum class ErrorSeverity {
    DEBUG,      ///< Debug information
    INFO,       ///< Informational message
    WARNING,    ///< Warning (non-critical)
    ERROR,      ///< Error (operation failed)
    CRITICAL,   ///< Critical error (system integrity at risk)
    FATAL       ///< Fatal error (system cannot continue)
};

/**
 * @brief Error categories for the framework
 */
enum class ErrorCategory {
    GENERAL,
    MEMORY,             ///< Memory-related errors
    COMPUTATION,        ///< Computation-related errors
    RADIATION,          ///< Radiation-induced errors
    CONFIGURATION,      ///< Configuration errors
    RESOURCE,           ///< Resource allocation errors
    THREADING,          ///< Threading or synchronization errors
    INITIALIZATION,     ///< Initialization errors
    IO,                 ///< I/O errors
    VALIDATION,         ///< Input validation errors
    INTERNAL,           ///< Internal framework errors
    EXTERNAL,           ///< Errors in external components
    TMR,                ///< TMR-related errors
    NEURAL_NETWORK,    ///< Neural network-related errors
    NETWORK             ///< Network-related errors
};

/**
 * @brief Error codes for the framework
 */
enum class ErrorCode {
    SUCCESS = 0,
    GENERAL_ERROR,
    INVALID_ARGUMENT,
    OUT_OF_MEMORY,
    BUFFER_OVERFLOW,
    INVALID_STATE,
    OPERATION_FAILED,
    NOT_IMPLEMENTED,
    TIMEOUT,
    IO_ERROR,
    MEMORY_ERROR,
    RADIATION_ERROR,
    NETWORK_ERROR,
    MISSING_DATA
};

/**
 * @brief Structured error information
 */
struct ErrorInfo {
    ErrorCode code;                      ///< Error code
    ErrorCategory category;              ///< Error category
    ErrorSeverity severity;              ///< Error severity
    std::string message;                 ///< Human-readable error message
    SourceLocation location;             ///< Source location where the error occurred
    std::optional<std::string> details;  ///< Additional details (optional)
    
    /**
     * @brief Constructor for ErrorInfo
     * 
     * @param code Error code
     * @param category Error category
     * @param severity Error severity
     * @param message Human-readable error message
     * @param location Source location where the error occurred
     * @param details Additional details (optional)
     */
    ErrorInfo(
        ErrorCode code,
        ErrorCategory category,
        ErrorSeverity severity,
        std::string message,
        SourceLocation location = RAD_ML_CURRENT_LOCATION,
        std::optional<std::string> details = std::nullopt)
        : code(code)
        , category(category)
        , severity(severity)
        , message(std::move(message))
        , location(location)
        , details(std::move(details)) {}
    
    /**
     * @brief Get string representation of the error
     * 
     * @param include_location Whether to include source location
     * @param include_details Whether to include detailed information
     * @return String representation of the error
     */
    std::string toString(bool include_location = true, bool include_details = true) const {
        std::stringstream ss;
        
        // Add severity and category
        ss << "[" << getSeverityString() << "][" << getCategoryString() << "] ";
        
        // Add error code and message
        ss << "Error " << static_cast<int>(code) << ": " << message;
        
        // Add source location if requested
        if (include_location) {
            ss << " (at " << location.file_name << ":" << location.line << ")";
        }
        
        // Add details if available and requested
        if (include_details && details) {
            ss << "\nDetails: " << *details;
        }
        
        return ss.str();
    }
    
    /**
     * @brief Get string representation of severity
     * 
     * @return String representation of severity
     */
    std::string getSeverityString() const {
        switch (severity) {
            case ErrorSeverity::DEBUG:    return "DEBUG";
            case ErrorSeverity::INFO:     return "INFO";
            case ErrorSeverity::WARNING:  return "WARNING";
            case ErrorSeverity::ERROR:    return "ERROR";
            case ErrorSeverity::CRITICAL: return "CRITICAL";
            case ErrorSeverity::FATAL:    return "FATAL";
            default:                      return "UNKNOWN";
        }
    }
    
    /**
     * @brief Get string representation of category
     * 
     * @return String representation of category
     */
    std::string getCategoryString() const {
        switch (category) {
            case ErrorCategory::GENERAL: return "GENERAL";
            case ErrorCategory::MEMORY:        return "MEMORY";
            case ErrorCategory::COMPUTATION:   return "COMPUTATION";
            case ErrorCategory::RADIATION:     return "RADIATION";
            case ErrorCategory::CONFIGURATION: return "CONFIGURATION";
            case ErrorCategory::RESOURCE:      return "RESOURCE";
            case ErrorCategory::THREADING:     return "THREADING";
            case ErrorCategory::INITIALIZATION: return "INITIALIZATION";
            case ErrorCategory::IO:            return "IO";
            case ErrorCategory::VALIDATION:    return "VALIDATION";
            case ErrorCategory::INTERNAL:      return "INTERNAL";
            case ErrorCategory::EXTERNAL:      return "EXTERNAL";
            case ErrorCategory::TMR:           return "TMR";
            case ErrorCategory::NEURAL_NETWORK: return "NEURAL_NETWORK";
            case ErrorCategory::NETWORK:       return "NETWORK";
            default:                           return "UNKNOWN";
        }
    }
};

/**
 * @brief Exception class for framework errors
 */
class RadiationFrameworkException : public std::exception {
private:
    ErrorInfo error_info_;
    mutable std::string message_cache_;
    
public:
    /**
     * @brief Constructor from ErrorInfo
     * 
     * @param error_info Error information
     */
    explicit RadiationFrameworkException(ErrorInfo error_info)
        : error_info_(std::move(error_info)) {}
    
    /**
     * @brief Constructor with individual parameters
     * 
     * @param code Error code
     * @param category Error category
     * @param severity Error severity
     * @param message Human-readable error message
     * @param location Source location where the error occurred
     * @param details Additional details (optional)
     */
    RadiationFrameworkException(
        ErrorCode code,
        ErrorCategory category,
        ErrorSeverity severity,
        std::string message,
        SourceLocation location = RAD_ML_CURRENT_LOCATION,
        std::optional<std::string> details = std::nullopt)
        : error_info_(code, category, severity, std::move(message), location, std::move(details)) {}
    
    /**
     * @brief Get error information
     * 
     * @return Error information
     */
    const ErrorInfo& getErrorInfo() const {
        return error_info_;
    }
    
    /**
     * @brief Get error code
     * 
     * @return Error code
     */
    ErrorCode getErrorCode() const {
        return error_info_.code;
    }
    
    /**
     * @brief Get error message
     * 
     * @return Error message
     */
    const char* what() const noexcept override {
        if (message_cache_.empty()) {
            message_cache_ = error_info_.toString();
        }
        return message_cache_.c_str();
    }
};

/**
 * @brief Result class template for operations that can fail
 * 
 * This class is used to return results from operations that can fail.
 * It contains either a value of type T or an ErrorInfo instance.
 * 
 * @tparam T Type of the value
 */
template <typename T>
class Result {
public:
    /**
     * @brief Default constructor
     * 
     * Creates a success result with default-constructed value
     */
    Result() : variant_(T()) {}
    
    /**
     * @brief Constructor from value
     * 
     * Creates a success result with the given value
     * 
     * @param value Value to store
     */
    Result(T value) : variant_(std::move(value)) {}
    
    /**
     * @brief Constructor from ErrorInfo
     * 
     * Creates an error result with the given error information
     * 
     * @param error Error information
     */
    Result(ErrorInfo error) : variant_(std::move(error)) {}
    
    /**
     * @brief Check if result is a success
     * 
     * @return True if result is a success, false if it's an error
     */
    bool isSuccess() const {
        return std::holds_alternative<T>(variant_);
    }
    
    /**
     * @brief Check if result is an error
     * 
     * @return True if result is an error, false if it's a success
     */
    bool isError() const {
        return std::holds_alternative<ErrorInfo>(variant_);
    }
    
    /**
     * @brief Get value (throws if error)
     * 
     * @return Reference to the value
     * @throws RadiationFrameworkException if result is an error
     */
    const T& value() const {
        if (isError()) {
            throw RadiationFrameworkException(error());
        }
        return std::get<T>(variant_);
    }
    
    /**
     * @brief Get value (throws if error)
     * 
     * @return Reference to the value
     * @throws RadiationFrameworkException if result is an error
     */
    T& value() {
        if (isError()) {
            throw RadiationFrameworkException(error());
        }
        return std::get<T>(variant_);
    }
    
    /**
     * @brief Get value or default
     * 
     * @param default_value Default value to return if result is an error
     * @return Value or default value
     */
    T valueOr(T default_value) const {
        if (isError()) {
            return default_value;
        }
        return std::get<T>(variant_);
    }
    
    /**
     * @brief Get error (throws if success)
     * 
     * @return Reference to the error information
     * @throws std::logic_error if result is a success
     */
    const ErrorInfo& error() const {
        if (!isError()) {
            throw std::logic_error("Result contains a value, not an error");
        }
        return std::get<ErrorInfo>(variant_);
    }
    
    /**
     * @brief Map success value to a new type
     * 
     * If this is a success, applies the function to the value and returns a new Result.
     * If this is an error, returns a new Result with the same error.
     * 
     * @tparam U Type of the return value of the function
     * @param f Function to apply to the value
     * @return Result of type U
     */
    template <typename U>
    Result<U> map(std::function<U(const T&)> f) const {
        if (isSuccess()) {
            return Result<U>(f(value()));
        } else {
            return Result<U>(error());
        }
    }
    
    /**
     * @brief Apply a function that returns a Result
     * 
     * If this is a success, applies the function to the value and returns the result.
     * If this is an error, returns a new Result with the same error.
     * 
     * @tparam U Type of the value in the result of the function
     * @param f Function to apply to the value
     * @return Result of type U
     */
    template <typename U>
    Result<U> flatMap(std::function<Result<U>(const T&)> f) const {
        if (isSuccess()) {
            return f(value());
        } else {
            return Result<U>(error());
        }
    }
    
    /**
     * @brief Factory method for success results
     * 
     * @tparam U Type of the value
     * @param value Value to store
     * @return Success result with the given value
     */
    template <typename U>
    static Result<U> success(U value) {
        return Result<U>(std::move(value));
    }
    
    /**
     * @brief Factory method for error results
     * 
     * @tparam U Type of the value
     * @param error Error information
     * @return Error result with the given error information
     */
    template <typename U>
    static Result<U> error(ErrorInfo error) {
        return Result<U>(std::move(error));
    }
    
    /**
     * @brief Factory method for error results with individual parameters
     * 
     * @tparam U Type of the value
     * @param code Error code
     * @param category Error category
     * @param severity Error severity
     * @param message Human-readable error message
     * @param location Source location where the error occurred
     * @param details Additional details (optional)
     * @return Error result with the given error information
     */
    template <typename U>
    static Result<U> error(
        ErrorCode code,
        ErrorCategory category,
        ErrorSeverity severity,
        std::string message,
        SourceLocation location = RAD_ML_CURRENT_LOCATION,
        std::optional<std::string> details = std::nullopt) {
        
        return Result<U>::error(
            ErrorInfo(code, category, severity, std::move(message), location, std::move(details)));
    }
    
private:
    std::variant<T, ErrorInfo> variant_;
};

/**
 * @brief Specialization of Result for void
 * 
 * This specialization is used for operations that don't return a value.
 */
template <>
class Result<void> {
public:
    /**
     * @brief Default constructor
     * 
     * Creates a success result
     */
    Result() : success_(true) {}
    
    /**
     * @brief Constructor from ErrorInfo
     * 
     * Creates an error result with the given error information
     * 
     * @param error Error information
     */
    Result(ErrorInfo error) : success_(false), error_(std::move(error)) {}
    
    /**
     * @brief Check if result is a success
     * 
     * @return True if result is a success, false if it's an error
     */
    bool isSuccess() const {
        return success_;
    }
    
    /**
     * @brief Check if result is an error
     * 
     * @return True if result is an error, false if it's a success
     */
    bool isError() const {
        return !success_;
    }
    
    /**
     * @brief Get error (throws if success)
     * 
     * @return Reference to the error information
     * @throws std::logic_error if result is a success
     */
    const ErrorInfo& error() const {
        if (!isError()) {
            throw std::logic_error("Result contains a value, not an error");
        }
        return *error_;
    }
    
    /**
     * @brief Factory method for success results
     * 
     * @return Success result
     */
    static Result<void> success() {
        return Result<void>();
    }
    
    /**
     * @brief Factory method for error results
     * 
     * @param error Error information
     * @return Error result with the given error information
     */
    static Result<void> error(ErrorInfo error) {
        return Result<void>(std::move(error));
    }
    
    /**
     * @brief Factory method for error results with individual parameters
     * 
     * @param code Error code
     * @param category Error category
     * @param severity Error severity
     * @param message Human-readable error message
     * @param location Source location where the error occurred
     * @param details Additional details (optional)
     * @return Error result with the given error information
     */
    static Result<void> error(
        ErrorCode code,
        ErrorCategory category,
        ErrorSeverity severity,
        std::string message,
        SourceLocation location = RAD_ML_CURRENT_LOCATION,
        std::optional<std::string> details = std::nullopt) {
        
        return Result<void>::error(
            ErrorInfo(code, category, severity, std::move(message), location, std::move(details)));
    }
    
private:
    bool success_;
    std::optional<ErrorInfo> error_;
};

/**
 * @brief Logger interface for error handling
 */
class IErrorLogger {
public:
    /**
     * @brief Virtual destructor
     */
    virtual ~IErrorLogger() = default;
    
    /**
     * @brief Log an error
     * 
     * @param error Error information
     */
    virtual void logError(const ErrorInfo& error) = 0;
    
    /**
     * @brief Log an error with parameters
     * 
     * @param code Error code
     * @param category Error category
     * @param severity Error severity
     * @param message Human-readable error message
     * @param location Source location where the error occurred
     * @param details Additional details (optional)
     */
    void logError(
        ErrorCode code,
        ErrorCategory category,
        ErrorSeverity severity,
        const std::string& message,
        const SourceLocation& location = RAD_ML_CURRENT_LOCATION,
        const std::optional<std::string>& details = std::nullopt) {
        
        logError(ErrorInfo(code, category, severity, message, location, details));
    }
};

/**
 * @brief Simple console logger implementation
 */
class ConsoleErrorLogger : public IErrorLogger {
public:
    /**
     * @brief Log an error to the console
     * 
     * @param error Error information
     */
    void logError(const ErrorInfo& error) override {
        std::string output = error.toString();
        
        // Use appropriate output stream based on severity
        if (error.severity == ErrorSeverity::DEBUG || 
            error.severity == ErrorSeverity::INFO || 
            error.severity == ErrorSeverity::WARNING) {
            std::cout << output << std::endl;
        } else {
            std::cerr << output << std::endl;
        }
    }
};

/**
 * @brief Error handler class
 */
class ErrorHandler {
private:
    static std::shared_ptr<IErrorLogger> logger_;
    static ErrorSeverity reporting_level_;
    
public:
    /**
     * @brief Set the logger
     * 
     * @param logger Logger to use
     */
    static void setLogger(std::shared_ptr<IErrorLogger> logger) {
        logger_ = std::move(logger);
    }
    
    /**
     * @brief Set the reporting level
     * 
     * Only errors with severity >= reporting_level will be logged
     * 
     * @param level Minimum severity level to log
     */
    static void setReportingLevel(ErrorSeverity level) {
        reporting_level_ = level;
    }
    
    /**
     * @brief Log an error
     * 
     * @param error Error information
     */
    static void logError(const ErrorInfo& error) {
        if (error.severity >= reporting_level_ && logger_) {
            logger_->logError(error);
        }
    }
    
    /**
     * @brief Handle an error
     * 
     * Logs the error and throws an exception if severity is high enough
     * 
     * @param error Error information
     * @param throw_on Minimum severity level to throw an exception
     * @throws RadiationFrameworkException if severity >= throw_on
     */
    static void handleError(
        const ErrorInfo& error,
        ErrorSeverity throw_on = ErrorSeverity::ERROR) {
        
        // Always log the error
        logError(error);
        
        // Throw an exception if severity is high enough
        if (error.severity >= throw_on) {
            throw RadiationFrameworkException(error);
        }
    }
    
    /**
     * @brief Handle an error result
     * 
     * If the result is an error, handles it according to the given parameters.
     * 
     * @tparam T Type of the value in the result
     * @param result Result to handle
     * @param throw_on Minimum severity level to throw an exception
     * @return True if result is a success, false if it's an error
     * @throws RadiationFrameworkException if severity >= throw_on
     */
    template <typename T>
    static bool handleErrorResult(
        const Result<T>& result,
        ErrorSeverity throw_on = ErrorSeverity::ERROR) {
        
        if (result.isError()) {
            handleError(result.error(), throw_on);
            return false;
        }
        
        return true;
    }
};

// Initialize static members
std::shared_ptr<IErrorLogger> ErrorHandler::logger_ = std::make_shared<ConsoleErrorLogger>();
ErrorSeverity ErrorHandler::reporting_level_ = ErrorSeverity::INFO;

// Assertion macro for invariant checks
#define RAD_ML_ASSERT(condition, message) \
    do { \
        if (!(condition)) { \
            rad_ml::error::ErrorHandler::handleError( \
                rad_ml::error::ErrorInfo( \
                    rad_ml::error::ErrorCode::VALIDATION_INVALID_OPERATION, \
                    rad_ml::error::ErrorCategory::VALIDATION, \
                    rad_ml::error::ErrorSeverity::FATAL, \
                    message \
                ) \
            ); \
        } \
    } while (0)

/**
 * @brief Type for error callback functions
 */
using ErrorCallback = std::function<void(const ErrorInfo&)>;

/**
 * @brief Singleton class for handling errors throughout the framework
 */
class ErrorHandler {
public:
    /**
     * @brief Get the singleton instance
     * 
     * @return Reference to the singleton instance
     */
    static ErrorHandler& getInstance() {
        static ErrorHandler instance;
        return instance;
    }
    
    /**
     * @brief Log an error using the registered callbacks
     * 
     * @param error Error information
     */
    static void logError(const ErrorInfo& error) {
        getInstance().logErrorImpl(error);
    }
    
    /**
     * @brief Register a callback for error handling
     * 
     * @param callback Function to call when an error occurs
     * @return ID of the registered callback
     */
    int registerCallback(ErrorCallback callback) {
        std::lock_guard<std::mutex> lock(mutex_);
        callbacks_.push_back(callback);
        return static_cast<int>(callbacks_.size() - 1);
    }
    
    /**
     * @brief Unregister a callback
     * 
     * @param id ID of the callback to unregister
     * @return true if the callback was found and unregistered
     */
    bool unregisterCallback(int id) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (id >= 0 && id < static_cast<int>(callbacks_.size())) {
            callbacks_[id] = nullptr;  // Mark as removed
            return true;
        }
        return false;
    }
    
    /**
     * @brief Get a string representation of an error code
     * 
     * @param code Error code
     * @return String representation
     */
    static std::string getErrorCodeString(ErrorCode code) {
        switch (code) {
            case ErrorCode::SUCCESS: return "SUCCESS";
            case ErrorCode::GENERAL_ERROR: return "GENERAL_ERROR";
            case ErrorCode::INVALID_ARGUMENT: return "INVALID_ARGUMENT";
            case ErrorCode::OUT_OF_MEMORY: return "OUT_OF_MEMORY";
            case ErrorCode::BUFFER_OVERFLOW: return "BUFFER_OVERFLOW";
            case ErrorCode::INVALID_STATE: return "INVALID_STATE";
            case ErrorCode::OPERATION_FAILED: return "OPERATION_FAILED";
            case ErrorCode::NOT_IMPLEMENTED: return "NOT_IMPLEMENTED";
            case ErrorCode::TIMEOUT: return "TIMEOUT";
            case ErrorCode::IO_ERROR: return "IO_ERROR";
            case ErrorCode::MEMORY_ERROR: return "MEMORY_ERROR";
            case ErrorCode::RADIATION_ERROR: return "RADIATION_ERROR";
            case ErrorCode::NETWORK_ERROR: return "NETWORK_ERROR";
            case ErrorCode::MISSING_DATA: return "MISSING_DATA";
            default: return "UNKNOWN_ERROR";
        }
    }
    
    /**
     * @brief Get a string representation of an error category
     * 
     * @param category Error category
     * @return String representation
     */
    static std::string getErrorCategoryString(ErrorCategory category) {
        switch (category) {
            case ErrorCategory::GENERAL: return "GENERAL";
            case ErrorCategory::MEMORY: return "MEMORY";
            case ErrorCategory::COMPUTATION: return "COMPUTATION";
            case ErrorCategory::RADIATION: return "RADIATION";
            case ErrorCategory::CONFIGURATION: return "CONFIGURATION";
            case ErrorCategory::RESOURCE: return "RESOURCE";
            case ErrorCategory::THREADING: return "THREADING";
            case ErrorCategory::INITIALIZATION: return "INITIALIZATION";
            case ErrorCategory::IO: return "IO";
            case ErrorCategory::VALIDATION: return "VALIDATION";
            case ErrorCategory::INTERNAL: return "INTERNAL";
            case ErrorCategory::EXTERNAL: return "EXTERNAL";
            case ErrorCategory::TMR: return "TMR";
            case ErrorCategory::NEURAL_NETWORK: return "NEURAL_NETWORK";
            case ErrorCategory::NETWORK: return "NETWORK";
            default: return "UNKNOWN_CATEGORY";
        }
    }
    
    /**
     * @brief Get a string representation of an error severity level
     * 
     * @param severity Error severity
     * @return String representation
     */
    static std::string getErrorSeverityString(ErrorSeverity severity) {
        switch (severity) {
            case ErrorSeverity::DEBUG: return "DEBUG";
            case ErrorSeverity::INFO: return "INFO";
            case ErrorSeverity::WARNING: return "WARNING";
            case ErrorSeverity::ERROR: return "ERROR";
            case ErrorSeverity::CRITICAL: return "CRITICAL";
            case ErrorSeverity::FATAL: return "FATAL";
            default: return "UNKNOWN_SEVERITY";
        }
    }

private:
    // Private constructor for singleton
    ErrorHandler() {
        // Register default console logger
        registerCallback([](const ErrorInfo& error) {
            std::cerr << "[" << getErrorSeverityString(error.severity) << "] "
                      << getErrorCategoryString(error.category) << " - "
                      << getErrorCodeString(error.code) << ": "
                      << error.message;
            
            if (!error.location.empty()) {
                std::cerr << " (" << error.location << ")";
            }
            
            std::cerr << std::endl;
        });
    }
    
    // Private implementation of logError
    void logErrorImpl(const ErrorInfo& error) {
        std::lock_guard<std::mutex> lock(mutex_);
        for (const auto& callback : callbacks_) {
            if (callback) {
                callback(error);
            }
        }
    }
    
    // Prevent copying and assignment
    ErrorHandler(const ErrorHandler&) = delete;
    ErrorHandler& operator=(const ErrorHandler&) = delete;
    
    std::vector<ErrorCallback> callbacks_;
    std::mutex mutex_;
};

} // namespace error
} // namespace rad_ml 