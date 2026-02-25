/**
 * Architecture & Design Decisions Page
 * - Mermaid.js initialization with project theme
 * - Scroll-triggered fade-in animations
 */

// Initialize Mermaid with project color scheme
mermaid.initialize({
    startOnLoad: true,
    theme: 'base',
    themeVariables: {
        primaryColor: '#1a1f2e',
        primaryTextColor: '#ffffff',
        primaryBorderColor: '#4BFF2A',
        lineColor: '#4BFF2A',
        secondaryColor: '#2A2AFF',
        tertiaryColor: '#1a1f2e',
        edgeLabelBackground: '#0f1117',
        noteBkgColor: '#F8F8FF',
        noteTextColor: '#1B2129',
        noteBorderColor: '#E2E4EA',
        actorBkg: '#2A2AFF',
        actorTextColor: '#ffffff',
        actorBorder: '#1a1acc',
        actorLineColor: '#06b6d4',
        signalColor: '#1B2129',
        signalTextColor: '#1B2129',
        labelBoxBkgColor: '#f0f0ff',
        labelBoxBorderColor: '#2A2AFF',
        labelTextColor: '#1B2129',
        loopTextColor: '#1B2129',
        activationBorderColor: '#2A2AFF',
        activationBkgColor: '#eff6ff',
        sequenceNumberColor: '#ffffff',
        fontFamily: "'Open Sans', sans-serif",
        fontSize: '9px',
    },
    flowchart: {
        useMaxWidth: true,
        htmlLabels: true,
        curve: 'basis',
        padding: 20,
        nodeSpacing: 30,
        rankSpacing: 50,
    },
    sequence: {
        useMaxWidth: true,
        actorMargin: 40,
        mirrorActors: false,
        messageMargin: 30,
        boxMargin: 8,
        noteMargin: 10,
    }
});

// Scroll-triggered fade-in for sections
document.addEventListener('DOMContentLoaded', function () {
    var sections = document.querySelectorAll('.fade-in-section');

    if ('IntersectionObserver' in window) {
        var observer = new IntersectionObserver(function (entries) {
            entries.forEach(function (entry) {
                if (entry.isIntersecting) {
                    entry.target.classList.add('visible');
                    observer.unobserve(entry.target);
                }
            });
        }, {
            threshold: 0.1,
            rootMargin: '0px 0px -40px 0px'
        });

        sections.forEach(function (section) {
            observer.observe(section);
        });
    } else {
        // Fallback: show all sections immediately
        sections.forEach(function (section) {
            section.classList.add('visible');
        });
    }
});
