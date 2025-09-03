"""Disease module system with auto-discovery."""

import importlib
import pkgutil
from typing import Dict, Type, List, Optional
import structlog
from .base import DiseaseModule, DrugClassConfig, DiseaseModuleConfig

logger = structlog.get_logger(__name__)


class DiseaseRegistry:
    """Registry for disease modules with auto-discovery."""
    
    def __init__(self):
        self._modules: Dict[str, Type[DiseaseModule]] = {}
        self._instances: Dict[str, DiseaseModule] = {}
        self._discover_modules()
    
    def _discover_modules(self) -> None:
        """Auto-discover disease modules in the diseases package."""
        # Get the path to this package
        package_path = __path__
        
        # Iterate through all modules in the diseases package
        for finder, module_name, ispkg in pkgutil.iter_modules(package_path):
            if module_name != 'base' and ispkg:  # Skip base module and only look at packages
                try:
                    # Import the module package
                    full_module_name = f"{__name__}.{module_name}.module"
                    module = importlib.import_module(full_module_name)
                    
                    # Look for MODULE_CLASS attribute
                    if hasattr(module, 'MODULE_CLASS'):
                        disease_class = getattr(module, 'MODULE_CLASS')
                        
                        # Verify it's a DiseaseModule subclass
                        if isinstance(disease_class, type) and issubclass(disease_class, DiseaseModule):
                            # Create an instance to get the name
                            instance = disease_class()
                            disease_name = instance.name
                            
                            # Store the class and instance
                            self._modules[disease_name] = disease_class
                            self._instances[disease_name] = instance
                            
                            logger.info(f"Discovered disease module", 
                                      disease=disease_name, 
                                      display_name=instance.display_name)
                        else:
                            logger.warning(f"MODULE_CLASS in {module_name} is not a DiseaseModule subclass")
                    else:
                        logger.debug(f"No MODULE_CLASS found in {module_name}")
                        
                except ImportError as e:
                    logger.error(f"Failed to import disease module", 
                               module=module_name, 
                               error=str(e))
                except Exception as e:
                    logger.error(f"Error discovering disease module", 
                               module=module_name, 
                               error=str(e))
    
    def get_module(self, name: str) -> Optional[DiseaseModule]:
        """
        Get disease module instance by name.
        
        Args:
            name: Disease module name (e.g., 'nsclc')
            
        Returns:
            DiseaseModule instance or None if not found
        """
        if name in self._instances:
            return self._instances[name]
        
        # Try to create a new instance if we have the class
        if name in self._modules:
            try:
                instance = self._modules[name]()
                self._instances[name] = instance
                return instance
            except Exception as e:
                logger.error(f"Failed to create disease module instance", 
                           disease=name, 
                           error=str(e))
        
        return None
    
    def list_available(self) -> List[str]:
        """
        List all available disease modules.
        
        Returns:
            List of disease module names
        """
        return list(self._modules.keys())
    
    def get_all_modules(self) -> Dict[str, DiseaseModule]:
        """
        Get all disease module instances.
        
        Returns:
            Dictionary mapping disease names to module instances
        """
        return self._instances.copy()
    
    def register_module(self, module_class: Type[DiseaseModule]) -> None:
        """
        Manually register a disease module.
        
        Args:
            module_class: DiseaseModule subclass to register
        """
        if not issubclass(module_class, DiseaseModule):
            raise TypeError(f"{module_class} must be a subclass of DiseaseModule")
        
        try:
            instance = module_class()
            disease_name = instance.name
            
            self._modules[disease_name] = module_class
            self._instances[disease_name] = instance
            
            logger.info(f"Manually registered disease module", 
                      disease=disease_name, 
                      display_name=instance.display_name)
        except Exception as e:
            logger.error(f"Failed to register disease module", 
                       module_class=module_class.__name__, 
                       error=str(e))
            raise
    
    def unregister_module(self, name: str) -> bool:
        """
        Unregister a disease module.
        
        Args:
            name: Disease module name to unregister
            
        Returns:
            True if module was unregistered, False if not found
        """
        if name in self._modules:
            del self._modules[name]
            if name in self._instances:
                del self._instances[name]
            logger.info(f"Unregistered disease module", disease=name)
            return True
        return False
    
    def reload_modules(self) -> None:
        """Reload all disease modules."""
        self._modules.clear()
        self._instances.clear()
        self._discover_modules()
        logger.info(f"Reloaded disease modules", count=len(self._modules))


# Global registry instance
disease_registry = DiseaseRegistry()

# Export key components
__all__ = [
    "DiseaseModule",
    "DrugClassConfig",
    "DiseaseModuleConfig",
    "DiseaseRegistry",
    "disease_registry",
]