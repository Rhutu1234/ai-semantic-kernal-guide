# Semantic Kernel

*A deep-dive walkthrough of Semantic Kernel — Microsoft's open-source, .NET-oriented SDK for building AI applications — covering the Kernel and its fundamentals, AI services and connectors, plugins and kernel functions, automatic function calling, prompt templates, memory (chat history and vector stores), agents, planning (and why function calling replaced the old planners), multi-agent orchestration, and implementing RAG with Semantic Kernel, plus where it fits alongside Microsoft.Extensions.AI and the Microsoft Agent Framework.*

---

## Table of Contents

1. [Introduction](#introduction)
2. [Semantic Kernel Fundamentals](#1-semantic-kernel-fundamentals)
3. [AI Services](#2-ai-services)
4. [Plugins](#3-plugins)
5. [Functions](#4-functions)
6. [Function Calling](#5-function-calling)
7. [Prompt Templates](#6-prompt-templates)
8. [Memory](#7-memory)
9. [Agents](#8-agents)
10. [Planning](#9-planning)
11. [Multi-Agent Systems](#10-multi-agent-systems)
12. [RAG with Semantic Kernel](#11-rag-with-semantic-kernel)
13. [Putting It Together: An ASP.NET Core Application](#12-putting-it-together-an-aspnet-core-application)
14. [Semantic Kernel vs. the Alternatives](#13-semantic-kernel-vs-the-alternatives)
15. [Common Pitfalls](#14-common-pitfalls)
16. [Quick Reference Table](#quick-reference-table)
17. [Conclusion](#conclusion)

---

## Introduction

Earlier guides in this series built AI features from the ground up: calling the API directly (*AI APIs with .NET*), designing the application around the model (*LLM Application Development*), and assembling retrieval pipelines by hand (*RAG*). That is valuable — you should understand the loop — but real applications keep re-implementing the same plumbing: a tool-calling loop, chat-history management, prompt templating, service configuration, logging, retries, and swapping model providers. **Semantic Kernel (SK)** packages that plumbing into a framework designed for .NET idioms: dependency injection, `IConfiguration`, `ILogger`, filters, and attributes.

```csharp
var builder = Kernel.CreateBuilder();
builder.AddOpenAIChatCompletion(modelId: "gpt-4o-mini", apiKey: apiKey);
builder.Plugins.AddFromType<WeatherPlugin>();
Kernel kernel = builder.Build();

var settings = new OpenAIPromptExecutionSettings { FunctionChoiceBehavior = FunctionChoiceBehavior.Auto() };

var result = await kernel.InvokePromptAsync("Do I need an umbrella in Pune today?", new(settings));
Console.WriteLine(result);     // the model calls WeatherPlugin as needed, then answers
```

Those few lines register a model, expose your C# method as a tool, and let the model decide when to call it — the entire tool-calling loop from this series' *AI APIs with .NET* guide is handled for you.

> **Read this first: Semantic Kernel moves fast.** SK is actively developed, and its API surface — especially around **agents**, **orchestration**, **memory/vector stores**, and embeddings — has been renamed and reorganized across releases, with several areas marked experimental. Microsoft has also introduced the **Microsoft Agent Framework**, built by the teams behind Semantic Kernel and AutoGen as a newer foundation for agents and multi-agent workflows (Section 13). Treat the code in this guide as illustrating *concepts and shapes*, compile it against the package versions you install, consult the current documentation for exact names, and check the framework's current status and migration guidance before starting a long-lived project.

---

## 1. Semantic Kernel Fundamentals

### What Semantic Kernel is

Semantic Kernel is a lightweight, open-source SDK (C#, Python, Java) that sits between your application and AI models. It provides:

```plaintext
CONNECTORS     A uniform way to talk to many AI providers (OpenAI, Azure OpenAI, Ollama, Hugging Face,
               and others) for chat, embeddings, image, and audio.
PLUGINS        A way to expose your own C# code (and APIs) to the model as callable tools.
FUNCTION CALLING  Automatic orchestration of the model <-> tool loop.
PROMPT TEMPLATES  Reusable, parameterized prompts with variables and function calls.
MEMORY         Chat history management and vector-store abstractions for retrieval.
AGENTS         Higher-level abstractions for autonomous, tool-using assistants and multi-agent systems.
FILTERS        Interception points for logging, approval, redaction, and safety.
OBSERVABILITY  Logging, metrics, and OpenTelemetry tracing built in.
```

### The Kernel: a service container plus an orchestrator

The central object is the **`Kernel`**. Think of it as a **dependency-injection container specialized for AI**: it holds the AI services (chat model, embedding model), the plugins (your tools), and the filters, and it orchestrates invocations among them.

```plaintext
                          ┌───────────────────────── Kernel ─────────────────────────┐
  Your application  ────> │  AI services      Plugins            Filters   Logging    │ ────> Model
                          │  (chat,           (kernel functions: (approval, telemetry │       providers
                          │   embeddings...)   your C# tools)     redaction)          │
                          └────────────────────────────────────────────────────────────┘
```

```csharp
var builder = Kernel.CreateBuilder();                       // a builder, similar to a host/service builder

builder.AddOpenAIChatCompletion("gpt-4o-mini", apiKey);     // register an AI service
builder.Plugins.AddFromType<OrdersPlugin>("Orders");        // register a plugin
builder.Services.AddLogging(l => l.AddConsole());           // the standard .NET service collection is available

Kernel kernel = builder.Build();
```

### Core vocabulary

```plaintext
Kernel            The container/orchestrator holding services, plugins, and filters.
KernelFunction    One callable unit — either a native C# method or a prompt (Sections 4, 6).
KernelPlugin      A named group of related kernel functions (Section 3).
KernelArguments   The named inputs (and execution settings) passed to a function or prompt.
FunctionResult    What a function returned, with metadata (token usage, etc.).
ExecutionSettings Model parameters — temperature, max tokens, function-choice behavior (Section 2).
ChatHistory       The ordered list of system/user/assistant/tool messages in a conversation.
```

### Invoking things

```csharp
// A plain prompt with a variable
var arguments = new KernelArguments { ["topic"] = "dependency injection" };
FunctionResult result = await kernel.InvokePromptAsync("Explain {{$topic}} in one sentence.", arguments);
string text = result.GetValue<string>()!;

// A specific function by plugin and name
var status = await kernel.InvokeAsync("Orders", "get_order_status", new() { ["orderNumber"] = 10482 });
```

### The Kernel is cheap — create per request

```plaintext
A Kernel is a lightweight object. In ASP.NET Core, register it with services.AddKernel() (which makes it
TRANSIENT) so each request gets its own instance. That is what lets you attach request-specific plugins
(the current user, the current tenant) safely — never share one mutable kernel with per-user state across
concurrent requests. Kernel.Clone() copies a kernel when you need a variant.
```

### Semantic Kernel and Microsoft.Extensions.AI

Microsoft.Extensions.AI (`IChatClient`, `IEmbeddingGenerator`) is the low-level, provider-neutral abstraction layer (see *AI APIs with .NET*, Section 13). Semantic Kernel builds on it: SK adds orchestration on top — plugins, prompt templates, filters, memory, and agents. Recent SK versions can use `IChatClient`/`IEmbeddingGenerator` services directly, so the two compose rather than compete.

---

## 2. AI Services

### Connectors register models behind uniform interfaces

An **AI service** is a registered model endpoint. Your code depends on the interface (`IChatCompletionService`), not on a specific vendor.

```csharp
var builder = Kernel.CreateBuilder();

// OpenAI
builder.AddOpenAIChatCompletion(modelId: "gpt-4o-mini", apiKey: openAiKey);

// Azure OpenAI — uses the DEPLOYMENT name (see AI APIs with .NET, Section 3)
builder.AddAzureOpenAIChatCompletion(
    deploymentName: "my-gpt-deployment",
    endpoint: "https://my-resource.openai.azure.com/",
    apiKey: azureKey);                        // overloads also accept an Entra ID TokenCredential (keyless)

Kernel kernel = builder.Build();
```

### The services you'll use

```plaintext
Chat completion       IChatCompletionService  — conversational models (the workhorse)
Embeddings            Text-embedding generation for RAG and memory (registered via AddOpenAI...Embedding... methods;
                      the exact builder method names changed as SK moved to Microsoft.Extensions.AI's
                      IEmbeddingGenerator — check your version)
Text to image / audio Image generation, speech-to-text, text-to-speech connectors
```

### Using the chat service directly

```csharp
IChatCompletionService chat = kernel.GetRequiredService<IChatCompletionService>();

var history = new ChatHistory();
history.AddSystemMessage("You are a concise C# tutor.");
history.AddUserMessage("What is a record?");

ChatMessageContent reply = await chat.GetChatMessageContentAsync(history, kernel: kernel);
history.AddAssistantMessage(reply.Content!);                 // YOU keep the conversation (models are stateless)

// Streaming
await foreach (StreamingChatMessageContent chunk in chat.GetStreamingChatMessageContentsAsync(history, kernel: kernel))
    Console.Write(chunk.Content);
```

### Execution settings: model parameters per call

```csharp
var settings = new OpenAIPromptExecutionSettings
{
    Temperature = 0.2,
    MaxTokens = 500,
    TopP = 1.0,
    FunctionChoiceBehavior = FunctionChoiceBehavior.Auto(),    // let the model call plugins (Section 5)
};
```

Settings mean what they mean in *AI APIs with .NET*, Section 9: low temperature for factual tasks, a token cap to bound cost and length. Provider-specific settings classes exist (for OpenAI/Azure OpenAI, Ollama, Hugging Face, and others); check what your model supports, since some reasoning models restrict parameters like temperature.

### Multiple models and routing by `ServiceId`

```csharp
builder.AddOpenAIChatCompletion("gpt-4o",      apiKey, serviceId: "smart");   // strong, expensive
builder.AddOpenAIChatCompletion("gpt-4o-mini", apiKey, serviceId: "fast");    // cheap, quick

// Choose per invocation
var cheap = new OpenAIPromptExecutionSettings { ServiceId = "fast" };
await kernel.InvokePromptAsync("Classify this ticket: {{$text}}", new(cheap) { ["text"] = ticketText });
```

Registering several services under IDs is how SK supports the **model routing** pattern from *LLM Application Development*, Section 9: send easy work to a small model and hard work to a stronger one, without changing the calling code.

### Providers and local models

Beyond OpenAI and Azure OpenAI, SK offers connectors for other providers and for local models (for example through Ollama or ONNX). Connector maturity varies — some are stable, others are preview — so check the status of the one you plan to depend on.

### Structured output

The OpenAI connector can constrain responses to a C# type (via a `ResponseFormat` setting that accepts a type), generating the JSON Schema for you — the SK equivalent of structured outputs in *AI APIs with .NET*, Section 7. As always, a schema guarantees *shape*, not *truth*: validate the values.

---

## 3. Plugins

### A plugin is a group of capabilities you expose to the model

A **plugin** is a named collection of **kernel functions** — typically a C# class whose methods the model may call. This is how an LLM gets "hands": looking up data, calling your services, performing actions.

```csharp
using System.ComponentModel;
using Microsoft.SemanticKernel;

public class OrdersPlugin(IOrderService orders, ICurrentUser user)      // dependencies via normal DI
{
    [KernelFunction("get_order_status")]
    [Description("Gets the status of one of the CURRENT customer's orders, given its order number.")]
    public async Task<string> GetOrderStatusAsync(
        [Description("The order number, for example 10482")] int orderNumber,
        CancellationToken ct = default)
    {
        // Authorization is enforced HERE, in code, using the authenticated user — never trusted to the model
        var order = await orders.GetForUserAsync(user.Id, orderNumber, ct);

        return order is null
            ? $"No order {orderNumber} was found for this customer."
            : $"Order {order.Number}: {order.Status}. Last updated {order.UpdatedOn:d}.";
    }

    [KernelFunction("cancel_order")]
    [Description("Cancels one of the current customer's orders. This is irreversible.")]
    public async Task<string> CancelOrderAsync([Description("The order number")] int orderNumber, CancellationToken ct = default)
        => await orders.CancelAsync(user.Id, orderNumber, ct) ? "The order was cancelled." : "The order could not be cancelled.";
}
```

### Registering plugins

```csharp
// With the builder (the plugin is created via DI, so its constructor dependencies resolve)
builder.Plugins.AddFromType<OrdersPlugin>("Orders");

// Or add to an existing kernel
kernel.Plugins.AddFromType<OrdersPlugin>("Orders");

// An already-constructed instance (handy for per-request state)
kernel.Plugins.AddFromObject(new OrdersPlugin(orders, currentUser), "Orders");
```

### Where plugins come from

```plaintext
NATIVE CODE      C# classes with [KernelFunction] methods (above). The most common source.
PROMPTS          Prompt functions defined in code, YAML, or files (Section 6).
OPENAPI          Import a REST API from its OpenAPI/Swagger spec — each operation becomes a function
                 (kernel.ImportPluginFromOpenApiAsync...). Fast for existing APIs; review which
                 operations you actually expose.
MCP              Model Context Protocol servers can be exposed as plugins, letting the model use the
                 growing ecosystem of MCP tools (check current connector status).
BUILT-IN         Core plugins (time, text helpers, etc.) — convenient, but review what you expose.
```

### Designing plugins well

```plaintext
- THE DESCRIPTIONS ARE THE API. The model decides when and how to call a function from its name, its
  [Description], and its parameter descriptions. Write them like documentation for a new colleague:
  what it does, when to use it, what the arguments mean, and any limits.
- SMALL, FOCUSED, WELL-NAMED functions beat a few giant ones. Use clear verbs (get_order_status).
- LEAST PRIVILEGE. Expose only what the scenario needs. Every function you add is something a
  prompt-injected or confused model could call.
- SEPARATE READ FROM WRITE. Reads are low risk; writes and irreversible actions (cancel, send, delete, pay)
  deserve confirmation (Section 5) and clear warnings in their descriptions.
- RETURN SMALL, USEFUL TEXT. Whatever the function returns goes back into the model's context window and
  counts as tokens. Return the few fields needed, not an entire database record.
- ENFORCE AUTHORIZATION IN CODE using the authenticated user (as above). Never let the model supply
  "which user am I acting as" as a parameter.
- VALIDATE ARGUMENTS like any untrusted input — they come from a model steered by user text. Use parameterized
  queries; never concatenate them into SQL or shell commands.
- RETURN HELPFUL ERRORS AS TEXT ("No order 123 found for this customer") so the model can recover or explain,
  instead of throwing opaque exceptions.
```

---

## 4. Functions

### Kernel functions: the unit everything else is built from

A **`KernelFunction`** is a callable unit with a name, a description, typed parameters, and a result. There are two kinds, and both are invoked the same way and can both be offered to the model.

### Native functions — your C# code

```csharp
public class TextPlugin
{
    [KernelFunction("word_count")]
    [Description("Counts the words in a piece of text.")]
    public int WordCount([Description("The text to count")] string text)
        => text.Split(' ', StringSplitOptions.RemoveEmptyEntries).Length;
}
```

Method parameters can include a `Kernel` (to call other functions or services from inside), a `CancellationToken`, and a `KernelArguments`, none of which the model sees. Return types can be simple values, strings, or objects that SK serializes to JSON for the model.

### Prompt functions — a prompt that behaves like a function

```csharp
KernelFunction summarize = kernel.CreateFunctionFromPrompt(
    promptTemplate: "Summarize the following text in {{$sentences}} sentences:\n\n{{$input}}",
    functionName: "summarize",
    description: "Summarizes text to a requested number of sentences.");

var result = await kernel.InvokeAsync(summarize, new() { ["input"] = articleText, ["sentences"] = 2 });
```

A prompt function is *a prompt template plus execution settings, wrapped as a function*. That uniformity is powerful: because prompts and code are both kernel functions, they compose — a prompt can call native code, native code can call a prompt, and the model can be offered either as a tool.

### Calling functions directly vs. letting the model call them

```csharp
// 1. YOU decide (deterministic): call a function explicitly
var count = await kernel.InvokeAsync("Text", "word_count", new() { ["text"] = "one two three" });

// 2. The MODEL decides (function calling — Section 5): offered as tools; the model chooses
```

```plaintext
Prefer DIRECT, explicit calls when the sequence of steps is known — it is cheaper, faster, testable, and
predictable. Let the model choose tools only for the parts that genuinely need open-ended judgment.
(This is the "workflow vs. agent" distinction from LLM Application Development, Section 11.)
```

### Function results and metadata

```csharp
FunctionResult r = await kernel.InvokePromptAsync("Say hello");
string? text = r.GetValue<string>();
var metadata = r.Metadata;       // provider metadata, such as token usage — log it for cost tracking
```

---

## 5. Function Calling

### Semantic Kernel runs the tool-calling loop for you

In *AI APIs with .NET*, Section 6, you wrote the loop by hand: send tool definitions, receive tool calls, run them, send results back, repeat. **Semantic Kernel does exactly that automatically** when you enable function calling.

```plaintext
1. SK converts each registered kernel function (name, description, parameter schema from your C# types)
   into the tool definitions the model understands.
2. The model replies with a request to call a function (name + JSON arguments).
3. SK deserializes the arguments, invokes your C# method, and appends the result to the chat history.
4. SK calls the model again with the results — repeating until the model produces a final answer.
```

```csharp
var settings = new OpenAIPromptExecutionSettings
{
    FunctionChoiceBehavior = FunctionChoiceBehavior.Auto()      // offer all kernel functions; auto-invoke the chosen ones
};

var history = new ChatHistory("You are a helpful customer-support assistant for ACME.");
history.AddUserMessage("What's the status of order 10482?");

IChatCompletionService chat = kernel.GetRequiredService<IChatCompletionService>();
ChatMessageContent reply = await chat.GetChatMessageContentAsync(history, settings, kernel);   // pass the kernel!

Console.WriteLine(reply.Content);
```

Passing the `kernel` is what gives the chat service access to your plugins.

### Function choice behaviors

```plaintext
FunctionChoiceBehavior.Auto()       The model decides whether and which functions to call; SK invokes them
                                    automatically (the default pattern).
FunctionChoiceBehavior.Required()   The model MUST call a function (useful to force structured extraction
                                    or a specific action).
FunctionChoiceBehavior.None()       Functions are described to the m
