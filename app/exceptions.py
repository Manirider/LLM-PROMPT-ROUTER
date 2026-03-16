class PromptRouterError(Exception):

    def __init__(self, message: str = "An unexpected error occurred."):
        self.message = message
        super().__init__(self.message)


class ClassificationError(PromptRouterError):

    def __init__(self, message: str = "Failed to classify the user's intent."):
        super().__init__(message)


class RoutingError(PromptRouterError):

    def __init__(self, message: str = "Failed to route the message to an expert."):
        super().__init__(message)




class EmptyMessageError(PromptRouterError):

    def __init__(self, message: str = "Message cannot be empty."):
        super().__init__(message)


class LLMAPIError(PromptRouterError):
    def __init__(self, message: str = "LLM API error occurred."):
        super().__init__(message)
